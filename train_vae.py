import torch
import numpy as np
import torch.nn as nn
from vae import *
import config

def load_data(filenames):
    """
        Loads the rollouts from the environment (array of filenames)
        and splits them into batches of given size. rollouts come
        in sequence one after another
        Outputs trnsor of size (n_batches, batch_size, 180) where 180 is the dimensionality of the observation
    """
    lidar_stores = []
    for filename in filenames:
        rollouts = np.load(filename)
        lidar_store = rollouts['observation_store'][:, :180]
        lidar_stores.append(lidar_store)
    lidar_store = torch.tensor(np.concatenate(lidar_stores), dtype=torch.float32).to(config.device)
    return lidar_store


def save_processed_observations(model, lidar_store, save_filename):
    """
        Runs all rollout observations through the VAE, and saves the resulting latents
    """
    lidar_store_hat, mean, log_var = model(lidar_store)
    np.savez_compressed(save_filename, mean=mean.cpu().detach().numpy(),
                        log_var=log_var.cpu().detach().numpy())

def save_processed_environmental_variables(filenames, save_filename):
    """
        From the rollouts, extracts the (1) state variables, (2) action sequences, and (3) end termination flags
        for every timestep, and saves them separately in the file save_filename
    """
    state_vars_store, action_store, end_flag_store = [], [], []
    for filename in filenames:
        rollouts = np.load(filename)
        state_vars_store.append(rollouts['observation_store'][:, 180:])
        action_store.append(rollouts['action_store'])
        end_flag_store.append(rollouts['end_flag_store'])
    state_vars_store = np.concatenate(state_vars_store)
    action_store = np.concatenate(action_store)
    end_flag_store = np.concatenate(end_flag_store)
    np.savez_compressed(save_filename, end_flag_store=end_flag_store,
                        action_store=action_store, state_vars_store=state_vars_store)


def train_vae(model, lidar_store_batches, optimizer, epochs, verbose=False):
    n_batches, batch_size, _ = lidar_store_batches.shape
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx in range(n_batches):
            x = lidar_store_batches[batch_idx]
            optimizer.zero_grad()
            x_hat, mean, log_var = model(x)
            loss = loss_elbo(x, x_hat, mean, log_var)
            overall_loss += loss.item()
            loss.backward()
            optimizer.step()
        if verbose and (epoch+1)%5 ==0:
            print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss / (batch_idx * batch_size))
    return overall_loss


if __name__=="__main__":
    print(f"Using {config.device}")
    # filenames = [f"data/rollouts_{config.rollouts_random_n}_random.npz",
    #              f"data/rollouts_{config.rollouts_dqn_n}_dqn.npz"]
    filenames = [f"data/rollouts_{config.rollouts_dqn_n}_dqn_{i}.npz" for i in range(5)]  # only train on DQN
    lidar_store = load_data(filenames)

    # batching data
    batch_size = 512
    n_batches = len(lidar_store) // batch_size
    total_data_len = n_batches * batch_size
    lidar_store_batches = lidar_store[:total_data_len].reshape(n_batches, batch_size, 180)

    # training VAE
    model = VAE(latent_dim=config.vae_latent_dim).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_vae(model, lidar_store_batches, optimizer, epochs=100, verbose=True)

    # saving model and latents
    torch.save(model.state_dict(), f"data/vae_model_{config.vae_latent_dim}dimlatent.pt")
    save_processed_observations(model, lidar_store, f"data/vae_preprocessed_{config.vae_latent_dim}dimlatent.npz")
    save_processed_environmental_variables(filenames, f"data/vae_rollouts_env_vars.npz")