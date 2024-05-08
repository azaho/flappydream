import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae_latent_dim = 8

rollouts_random_n = 2000
rollouts_dqn_n = 2000

state_variables_names =  [
    "the last pipe's horizontal position",
    "the last top pipe's vertical position",
    "the last bottom pipe's vertical position",
    "the next pipe's horizontal position",
    "the next top pipe's vertical position",
    "the next bottom pipe's vertical position",
    "the next next pipe's horizontal position",
    "the next next top pipe's vertical position",
    "the next next bottom pipe's vertical position",
    "player's vertical position",
    "player's vertical velocity",
    "player's rotation",
]