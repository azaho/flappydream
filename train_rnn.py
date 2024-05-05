import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy
from rnn import *
from vae import *
import config
from pathlib import Path
import json
import time
import argparse
import collections
import os


def load_data(filename_vae_latents, filename_environment_vars, batch_size=512, truncate_at_batch=None):
    """
        Loads all the VAE-preprocessed latents from rollouts from filename_vae_latents
        and the corresponding environmental variables (actions etc) from filename_environment_vars,
        and splits them into batches of size batch_size of separate complete episodes.
        Returns all of this data
        truncate_at_batch: whether to load all data or a subset (truncate_at_batch batches)
    """
    # Load data from files
    z = np.load(filename_vae_latents)
    mean_store = torch.tensor(z['mean'])
    var_store = torch.tensor(z['log_var'])
    env_meta = np.load(filename_environment_vars)
    state_vars = torch.tensor(env_meta['state_vars_store'])
    end_flag_store = torch.tensor(env_meta['end_flag_store'])
    action_store = torch.tensor(env_meta['action_store'])
    # Get the number of episodes
    num_episodes = (end_flag_store).sum().item()
    # Initialize empty lists to store the batches
    mean_batches = []
    var_batches = []
    state_vars_batches = []
    action_batches = []
    end_flag_batches = []
    mask_batches = []
    # Iterate over the episodes and create batches
    nonzero_end_flag_store = torch.nonzero(end_flag_store)
    for i in range(0, num_episodes, batch_size):
        # Get the current batch of episodes
        batch_indices = range(i, min(i + batch_size, num_episodes))
        # Find the maximum episode length for the current batch
        max_episode_length = 0
        for idx in batch_indices:
            episode_start = nonzero_end_flag_store[idx-1] if idx > 0 else 0
            episode_end = nonzero_end_flag_store[idx] + 1
            episode_length = episode_end - episode_start
            max_episode_length = max(max_episode_length, episode_length.item())
        # Create empty tensors for the current batch
        mean_batch = torch.zeros((batch_size, max_episode_length, config.vae_latent_dim)).to(config.device)
        var_batch = torch.zeros((batch_size, max_episode_length, config.vae_latent_dim)).to(config.device)
        state_vars_batch = torch.zeros((batch_size, max_episode_length, 12)).to(config.device)
        action_batch = torch.zeros((batch_size, max_episode_length, 1)).to(config.device)
        end_flag_batch = torch.zeros((batch_size, max_episode_length, 1)).to(config.device)
        mask_batch = torch.zeros((batch_size, max_episode_length)).to(config.device)
        # Fill the tensors with data from the current batch of episodes
        for j, idx in enumerate(batch_indices):
            episode_start = nonzero_end_flag_store[idx-1] if idx > 0 else 0
            episode_end = nonzero_end_flag_store[idx] + 1
            episode_length = episode_end - episode_start

            mean_batch[j, :episode_length] = mean_store[episode_start:episode_end]
            var_batch[j, :episode_length] = var_store[episode_start:episode_end]
            state_vars_batch[j, :episode_length] = state_vars[episode_start:episode_end]
            action_batch[j, :episode_length] = action_store[episode_start:episode_end, None]
            mask_batch[j, :episode_length] = 1
            end_flag_batch[j, episode_length-1] = 1
        # Append the current batch to the lists
        mean_batches.append(mean_batch)
        var_batches.append(var_batch)
        state_vars_batches.append(state_vars_batch)
        action_batches.append(action_batch)
        mask_batches.append(mask_batch)
        end_flag_batches.append(end_flag_batch)
        if truncate_at_batch is not None and len(mean_batches) == truncate_at_batch: break
    n_batches = len(mean_batches)
    return n_batches, mean_batches, var_batches, state_vars_batches, action_batches, mask_batches, end_flag_batches


def train_rnn(model, training_data, n_epochs, optimizer, save_every_epochs=50, verbose=False, rnn_id=0, note_every_epochs=10, save_folder="data", detach_gradients=True, max_gradient_norm=None):
    """
        Trains the RNN
    """
    n_batches, mean_batches, var_batches, state_vars_batches, \
        action_batches, mask_batches, end_flag_batches = training_data
    batch_size = mean_batches[0].shape[0]
    losses_store = np.zeros((n_epochs, n_batches, 4))
    losses_store[:] = np.nan
    gradient_norms_store = np.zeros((n_epochs, n_batches, 1))
    gradient_norms_store[:] = np.nan

    # Configure logging
    print(f"Using {config.device}") # TODO: fix logger
    Path(f"{save_folder}/rnn{rnn_id}/").mkdir(parents=True, exist_ok=True)

    # Check for the last saved epoch
    last_epoch = 0
    for file in os.listdir(f"{save_folder}/rnn{rnn_id}/"):
        if file.startswith("rnn_model_epoch") and file.endswith(".pt"):
            epoch_num = int(file.split("epoch")[1].split(".")[0])
            if epoch_num > last_epoch:
                last_epoch = epoch_num
    if last_epoch > 0:
        print(f"Resuming training from epoch {last_epoch}")
        model.load_state_dict(torch.load(f"{save_folder}/rnn{rnn_id}/rnn_model_epoch{last_epoch}.pt"))
        optimizer.load_state_dict(torch.load(f"{save_folder}/rnn{rnn_id}/rnn_optimizer_epoch{last_epoch}.pt"))
        losses_store = np.load(f"{save_folder}/rnn{rnn_id}/rnn_losses.npz")["losses_store"]
        gradient_norms_store = np.load(f"{save_folder}/rnn{rnn_id}/rnn_gradientnorms.npz")["losses_store"]

    noted_time = time.time()
    epoch_states = []
    exceptions = []
    epoch = last_epoch
    restored=False
    while epoch<n_epochs:
        # Set initial hidden and cell states
        for batch_i in range(n_batches):
            hidden = model.init_hidden(batch_size)

            means = mean_batches[batch_i]
            vars = var_batches[batch_i]
            actions = action_batches[batch_i]
            masks = mask_batches[batch_i][:, 1:]
            end_flags = end_flag_batches[batch_i]
            state_vars = state_vars_batches[batch_i][:, :-1]
            z_values = reparameterization(means, vars)
            # z_values=means
            # inputs = z(t) + action(t)
            inputs = torch.cat([z_values[:, :-1], actions[:, :-1]], dim=2)  # [:, :-1]
            # targets = z(t+1) + end_flag(t+1)
            targets = torch.cat([z_values, end_flags], dim=2)[:, 1:]

            try:
                if detach_gradients: hidden = detach(hidden)
                (pi, mu, sigma), ef, hidden, y = model(inputs, hidden)

                loss_mdn = loss_pred(targets, pi, mu, sigma, masks)
                loss_ef = loss_errorflag(targets, ef, masks) * 10
                if model.n_state_vars > 0:
                    loss_sv = loss_statevars(state_vars[:, :, model.state_vars_to_predict], model.get_decoded_state_vars(y), masks) * 10
                else:
                    loss_sv = torch.tensor(0)
                loss = loss_mdn + loss_ef + loss_sv
                losses_store[epoch, batch_i, :] = [loss_mdn.item(), loss_ef.item(), loss_sv.item(), loss.item()]

                # Backward and optimize
                model.zero_grad()
                loss.backward()

                if max_gradient_norm is not None: torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm) # clip in training
                grad_norm = np.sqrt(sum([torch.norm(p.grad) ** 2 for p in model.parameters()]).item())
                gradient_norms_store[epoch, batch_i, 0] = grad_norm
                optimizer.step()

            except Exception as e:
                print(f"Exception occurred at epoch {epoch+1}, batch {batch_i+1}: {str(e)}")
                print("Restoring model from the last known good checkpoint.")
                exceptions.append((epoch+1, batch_i+1))

                # Clear GPU memory
                torch.cuda.empty_cache()

                # Move model and optimizer to CPU
                model.to('cpu')
                optimizer_state = optimizer.state_dict()
                optimizer.state = collections.defaultdict(dict)
                optimizer.load_state_dict(optimizer_state)

                # Restore model and optimizer states
                checkpoint = epoch_states[0]
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                # Move model and optimizer back to GPU
                model.to(config.device)
                optimizer_state = optimizer.state_dict()
                optimizer.state = collections.defaultdict(dict)
                optimizer.load_state_dict(optimizer_state)

                noted_time = time.time()  # Reset the timer
                restored=True
                break
        if restored:
            restored=False
            continue

        # Save model and optimizer state every epoch
        epoch_states.append({
            'epoch': epoch,
            'model_state_dict': copy.deepcopy(model.state_dict()),
            'optimizer_state_dict': copy.deepcopy(optimizer.state_dict())
        })
        while len(epoch_states) > 20:
            epoch_states.pop(0)

        if ((epoch+1) % note_every_epochs == 0):
            epoch_end_time = time.time()
            elapsed_time = epoch_end_time - noted_time
            noted_time = epoch_end_time
            epochs_remaining = n_epochs - epoch - 1
            eta = elapsed_time / note_every_epochs * epochs_remaining
            hours, rem = divmod(eta, 3600)
            minutes, seconds = divmod(rem, 60)
            print('Epoch [{}/{}], Loss: {:.4f} ({:.4f}, {:.4f}, {:.4f})'
                  .format(epoch+1, n_epochs, loss.item(), loss_mdn.item(), loss_ef.item(), loss_sv.item()))
            print('     ETA: {:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds))
        if (epoch+1) % save_every_epochs == 0 or (epoch+1)==n_epochs:
            torch.save(model.state_dict(), f"{save_folder}/rnn{rnn_id}/rnn_model_epoch{epoch+1}.pt")
            torch.save(optimizer.state_dict(), f"{save_folder}/rnn{rnn_id}/rnn_optimizer_epoch{epoch+1}.pt")
            np.savez_compressed(f"{save_folder}/rnn{rnn_id}/rnn_losses.npz", losses_store=losses_store)
            np.savez_compressed(f"{save_folder}/rnn{rnn_id}/rnn_gradientnorms.npz", losses_store=gradient_norms_store)
            with open(f"{save_folder}/rnn{rnn_id}/rnn_meta.json", "w") as out_file:
                json.dump({"state_vars_to_predict": model.state_vars_to_predict.tolist() if model.n_state_vars>0 else [],
                           "trained_epochs": n_epochs, "exceptions_log": exceptions
                           }, out_file, indent = 4)
            print("== NETWORK SAVED\n")
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MDNRNN model')
    parser.add_argument('--truncate_at_batch', type=int, default=2, help='Number of batches to truncate the data')
    parser.add_argument('--rnn_id', type=str, default="0", help='String ID of the RNN model')
    args = parser.parse_args()

    print(f"Using {config.device}")

    # step 1: load data
    dim_latent_z = config.vae_latent_dim
    training_data = load_data(f'data/vae_preprocessed_{dim_latent_z}dimlatent.npz', f'data/vae_rollouts_env_vars.npz', truncate_at_batch=args.truncate_at_batch)

    # step 2: define the model and parameters
    model = MDNRNN(dim_latent_z, state_vars_to_predict=[]).to(config.device)
    optimizer = torch.optim.Adam(model.parameters())  # , lr=0.001, weight_decay=0.001)

    train_rnn(model, training_data, 10, optimizer, save_every_epochs=1, note_every_epochs=1, verbose=True, rnn_id=args.rnn_id)