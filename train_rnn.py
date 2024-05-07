import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy
from rnn import *
from vae_old import *
import config
from pathlib import Path
import json
import time
import argparse
import collections
import os
import logging
import datetime


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
    sigma_store = torch.tensor(z['log_var'])
    env_meta = np.load(filename_environment_vars)
    state_vars = torch.tensor(env_meta['state_vars_store'])
    end_flag_store = torch.tensor(env_meta['end_flag_store'])
    action_store = torch.tensor(env_meta['action_store'])
    # Get the number of episodes
    num_episodes = (end_flag_store).sum().item()
    # Initialize empty lists to store the batches
    mean_batches = []
    sigma_batches = []
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
        sigma_batch = torch.zeros((batch_size, max_episode_length, config.vae_latent_dim)).to(config.device)
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
            sigma_batch[j, :episode_length] = sigma_store[episode_start:episode_end]
            state_vars_batch[j, :episode_length] = state_vars[episode_start:episode_end]
            action_batch[j, :episode_length] = action_store[episode_start:episode_end, None]
            mask_batch[j, :episode_length] = 1
            end_flag_batch[j, episode_length-1] = 1
        # Append the current batch to the lists
        mean_batches.append(mean_batch)
        sigma_batches.append(sigma_batch)
        state_vars_batches.append(state_vars_batch)
        action_batches.append(action_batch)
        mask_batches.append(mask_batch)
        end_flag_batches.append(end_flag_batch)
        if truncate_at_batch is not None and len(mean_batches) == truncate_at_batch: break
    n_batches = len(mean_batches)
    return n_batches, mean_batches, sigma_batches, state_vars_batches, action_batches, mask_batches, end_flag_batches


def train_rnn(model, training_data, n_epochs, optimizer, save_every_epochs=50, verbose=False, rnn_id=0,
              note_every_epochs=5, save_folder="data", detach_gradients=True, max_gradient_norm=None,
              lambda_ef=1, multiplier_ef=10, lambda_sv=10):
    """
        Trains the RNN

        multiplier_ef is the multiplier of the readout value (e.g. error flag = 10 instead of =1).
        Helps the network output larger values when it thinks the game is getting over.
    """
    n_batches, mean_batches, sigma_batches, state_vars_batches, \
        action_batches, mask_batches, end_flag_batches = training_data
    batch_size = mean_batches[0].shape[0]
    losses_store = np.zeros((n_epochs, n_batches, 4))
    losses_store[:] = np.nan
    gradient_norms_store = np.zeros((n_epochs, n_batches, 1))
    gradient_norms_store[:] = np.nan

    # Make dirs if they are missing
    Path(f"{save_folder}/rnn{rnn_id}/").mkdir(parents=True, exist_ok=True)

    # Set up logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        handlers=[
                            logging.FileHandler(f"{save_folder}/rnn{rnn_id}/training.log"),
                            logging.StreamHandler()
                        ])
    logging.info(f"Using {config.device}")

    # Check for the last saved epoch
    def _get_last_saved_epoch():
        last_epoch = 0
        for file in os.listdir(f"{save_folder}/rnn{rnn_id}/"):
            if file.startswith("rnn_model_epoch") and file.endswith(".pt"):
                epoch_num = int(file.split("epoch")[1].split(".")[0])
                if epoch_num > last_epoch:
                    last_epoch = epoch_num
        return last_epoch
    def _restore_from_saved_epoch(last_epoch):
        logging.info(f"Resuming training from epoch {last_epoch}")
        model.load_state_dict(torch.load(f"{save_folder}/rnn{rnn_id}/rnn_model_epoch{last_epoch}.pt"))
        optimizer.load_state_dict(torch.load(f"{save_folder}/rnn{rnn_id}/rnn_optimizer_epoch{last_epoch}.pt"))
        losses_store = np.load(f"{save_folder}/rnn{rnn_id}/rnn_losses.npz")["losses_store"]
        gradient_norms_store = np.load(f"{save_folder}/rnn{rnn_id}/rnn_gradientnorms.npz")["losses_store"]
    epoch = _get_last_saved_epoch()
    if epoch>0: _restore_from_saved_epoch(epoch)

    noted_time = time.time()
    epoch_states = []
    exceptions = []
    restored = False
    N_restored_in_a_row = 0  # Once many restored in a row, go to the last saved network
    batch_indices = np.arange(n_batches)
    while epoch<n_epochs:
        np.random.shuffle(batch_indices)  # reshuffle indices
        for batch_i in batch_indices:
            # Set initial hidden and cell states
            hidden = model.init_hidden(batch_size)

            means = mean_batches[batch_i]
            sigmas = sigma_batches[batch_i]
            actions = action_batches[batch_i]
            masks = mask_batches[batch_i][:, 1:]
            end_flags = end_flag_batches[batch_i]
            state_vars = state_vars_batches[batch_i][:, :-1]
            z_values = reparameterization(means, sigmas)
            # z_values=means
            # inputs = z(t) + action(t)
            inputs = torch.cat([z_values[:, :-1], actions[:, :-1]], dim=2)  # [:, :-1]
            # targets = z(t+1) + end_flag(t+1)
            targets = torch.cat([z_values, end_flags], dim=2)[:, 1:]

            try:
                if detach_gradients: hidden = detach(hidden)
                (pi, mu, sigma), ef, hidden, y = model(inputs, hidden)

                loss_mdn = loss_pred(targets, pi, mu, sigma, masks)
                loss_ef = loss_errorflag(targets, ef, masks, multiplier_ef) * lambda_ef
                if model.n_state_vars > 0:
                    loss_sv = loss_statevars(state_vars[:, :, model.state_vars_to_predict],
                                             model.get_decoded_state_vars(y), masks) * lambda_sv
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
                logging.info(f"Exception occurred at epoch {epoch+1}, batch {batch_i+1}. {str(e)[:500]}")
                logging.info("Restoring model from the last known good checkpoint.")
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

                epoch = checkpoint['epoch']+1
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                # Move model and optimizer back to GPU
                model.to(config.device)
                optimizer_state = optimizer.state_dict()
                optimizer.state = collections.defaultdict(dict)
                optimizer.load_state_dict(optimizer_state)

                noted_time = time.time()  # Reset the timer
                N_restored_in_a_row += 1
                restored=True
                break
        if restored:
            if N_restored_in_a_row>10:
                epoch = _get_last_saved_epoch()
                if epoch > 0: _restore_from_saved_epoch(epoch)
                N_restored_in_a_row = 0
            restored=False
            continue
        N_restored_in_a_row = 0

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
            logging.info('Epoch [{}/{}], Loss: {:.4f} ({:.4f}, {:.4f}, {:.4f})'
                  .format(epoch+1, n_epochs, loss.item(), loss_mdn.item(), loss_ef.item(), loss_sv.item()))
            logging.info('     ETA: {:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds))
            np.savez_compressed(f"{save_folder}/rnn{rnn_id}/rnn_losses.npz", losses_store=losses_store) # save losses more often
        if (epoch+1) % save_every_epochs == 0 or (epoch+1)==n_epochs:
            torch.save(model.state_dict(), f"{save_folder}/rnn{rnn_id}/rnn_model_epoch{epoch+1}.pt")
            torch.save(optimizer.state_dict(), f"{save_folder}/rnn{rnn_id}/rnn_optimizer_epoch{epoch+1}.pt")
            np.savez_compressed(f"{save_folder}/rnn{rnn_id}/rnn_losses.npz", losses_store=losses_store)
            np.savez_compressed(f"{save_folder}/rnn{rnn_id}/rnn_gradientnorms.npz", losses_store=gradient_norms_store)
            with open(f"{save_folder}/rnn{rnn_id}/rnn_meta.json", "w") as out_file:
                json.dump({"state_vars_to_predict": model.state_vars_to_predict.tolist() if model.n_state_vars>0 else [],
                           "trained_epochs": n_epochs, "exceptions_log": exceptions
                           }, out_file, indent = 4)
            logging.info("== NETWORK SAVED\n")
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MDNRNN model')
    parser.add_argument('--truncate_at_batch', '-t', type=int, default=40, help='Number of batches to truncate the data')
    parser.add_argument('--state_vars_to_predict', '-sv', type=int, nargs='+', default=[],
                        help='List of state variables to predict')
    parser.add_argument('--detach_gradients', '-dg', action='store_true', help='Whether to detach gradients')
    parser.add_argument('--use_layernorm', '-ln', action='store_true', help='Whether to use layer normalization')
    parser.add_argument('--n_hidden', '-nh', type=int, default=128, help='Number of hidden units in the RNN')
    parser.add_argument('--dim_latent_z', '-dlz', type=int, default=8, help='Dimensionality of the latent observation space of the VAE')
    parser.add_argument('--train_epochs', '-e', type=int, default=250, help='Number of training epochs')
    parser.add_argument('--save_every_epochs', '-se', type=int, default=50,
                        help='Save the model every specified number of epochs')
    parser.add_argument('--max_gradient_norm', '-mgn', type=float, default=100.0,
                        help='Maximum gradient norm for gradient clipping')
    parser.add_argument('--lr', '-lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--random_index', '-r', type=int, default=0, help='Random seed index')
    parser.add_argument('--lambda_sv', '-lsv', type=float, default=10.0, help='Lambda value for state variables')
    parser.add_argument('--enforce_cuda', '-ecuda', action='store_true', help='Exit if cuda is not available')

    args = parser.parse_args()
    state_vars_to_predict = args.state_vars_to_predict
    detach_gradients = args.detach_gradients
    use_layernorm = args.use_layernorm
    truncate_at_batch = args.truncate_at_batch
    n_hidden = args.n_hidden
    train_epochs = args.train_epochs
    save_every_epochs = args.save_every_epochs
    max_gradient_norm = args.max_gradient_norm
    lr = args.lr
    random_index = args.random_index
    lambda_sv = args.lambda_sv
    dim_latent_z = args.dim_latent_z
    if lambda_sv == 0: state_vars_to_predict = []

    if args.enforce_cuda and not torch.cuda.is_available():
        print("CUDA NOT AVAILABLE! Exiting.")
        exit()

    sv_str = 'x'.join([str(x) for x in state_vars_to_predict]) if len(state_vars_to_predict)>0 else 'X'
    rnn_id = f"_ln{1 if use_layernorm else 0}_nh{n_hidden}_dlz{dim_latent_z}_mgn{max_gradient_norm}_lr{lr}" + \
             f"_dg{1 if detach_gradients else 0}_da{truncate_at_batch}_sv{sv_str}_lsv{lambda_sv}_r{random_index}"
    print("ID: " + rnn_id)

    training_data = load_data(f'data/vae_preprocessed_{dim_latent_z}dimlatent.npz',
                              f'data/vae_rollouts_env_vars.npz',
                              truncate_at_batch=truncate_at_batch, batch_size=256)

    # step 2: define the model and parameters
    model = MDNRNN(dim_latent_z, use_layernorm=use_layernorm, state_vars_to_predict=state_vars_to_predict,
                   n_hidden=n_hidden).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_rnn(model, training_data, train_epochs, optimizer, save_every_epochs=save_every_epochs, verbose=True,
              save_folder="data", rnn_id=rnn_id, max_gradient_norm=max_gradient_norm, lambda_sv=lambda_sv,
              detach_gradients=detach_gradients)