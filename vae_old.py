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
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

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
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)
        z = mean + var * epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar

def reparameterization(mean, var):
    """
        Implementing the reparametrization trick for the VAE data
        (resampling all latent variables from the distribution
            provided by the VAE in order to make sure the RNN doesn't overfit
            to any particular sample; this is following the original World Models paper; https://worldmodels.github.io/)
    """
    epsilon = torch.randn_like(var).to(config.device)
    z = mean + var * epsilon
    return z

def loss_elbo(x, x_hat, mean, log_var):
    """
        SIKE! This is elbo loss, but in its use somehow
        sigma is passed instead of log var (bug).
        So, it's kind of a weird KLD term, which is equivalent to (mu^2 + 0.5*sigma^2)/2 up to O(sigma^6)
    """
    reproduction_loss = torch.sum((x_hat-x)**2)#nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD
