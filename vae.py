import torch
import numpy as np
import torch.nn as nn
import config

class VAE(nn.Module):

    def __init__(self, input_dim=180, hidden_dim=360, latent_dim=config.vae_latent_dim):
        super(VAE, self).__init__()
        self.device = config.device

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )

        # latent mean and variance
        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim) # not actually log var! instead, it's just sigma

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mean, sigma = self.mean_layer(x), self.logvar_layer(x)
        return mean, sigma

    def reparameterization(self, mean, sigma):
        epsilon = torch.randn_like(sigma).to(self.device)
        z = mean + sigma * epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, sigma = self.encode(x)
        z = self.reparameterization(mean, sigma)
        x_hat = self.decode(z)
        return x_hat, mean, sigma

def reparameterization(mean, sigma):
    """
        Implementing the reparametrization trick for the VAE data
        (resampling all latent variables from the distribution
            provided by the VAE in order to make sure the RNN doesn't overfit
            to any particular sample; this is following the original World Models paper; https://worldmodels.github.io/)
    """
    epsilon = torch.randn_like(sigma).to(config.device)
    z = mean + sigma * epsilon
    return z

def loss_elbo(x, x_hat, mean, sigma):
    """
        SIKE! This is not elbo loss. Instead, it's
        MSE + (mu^2 + sigma^2/2)/2
    """
    reproduction_loss = torch.sum((x_hat-x)**2)#nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    #KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    #KLD = - 0.5 * torch.sum(1 + sigma.pow(2).log() - mean.pow(2) - sigma.pow(2))
    KLD = 0.5 * torch.sum(mean.pow(2) + sigma.pow(2)/2)
    return reproduction_loss + KLD

