import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy
import config
import lstm

class MDNRNN(nn.Module):
    """
        MDN-RNN implementation
    """
    def __init__(self, z_size, n_hidden=128, n_gaussians=5, n_layers=1, state_vars_to_predict=None, use_layernorm=False):
        super(MDNRNN, self).__init__()

        self.z_size = z_size
        self.n_hidden = n_hidden
        self.n_gaussians = n_gaussians
        self.n_layers = n_layers

        self.state_vars_to_predict = [] if state_vars_to_predict is None or len(state_vars_to_predict)==0 else np.array(state_vars_to_predict)
        self.n_state_vars = len(self.state_vars_to_predict)

        if use_layernorm:
            self.lstm = lstm.LSTM(z_size+1, n_hidden, n_layers, batch_first=True, bidirectional=0, cln=False) # +1 for action input
        else:
            self.lstm = nn.LSTM(z_size+1, n_hidden, n_layers, batch_first=True) # +1 for action input
        self.fc1 = nn.Linear(n_hidden, n_gaussians)
        self.fc2 = nn.Linear(n_hidden, n_gaussians*z_size)
        self.fc3 = nn.Linear(n_hidden, n_gaussians*z_size)
        self.fc4 = nn.Linear(n_hidden, 1) # for predicting the end flag
        if self.n_state_vars>0:
            self.fc5 = nn.Linear(n_hidden, self.n_state_vars) # optional; for deconding n env vars

    def get_mixture_coef(self, y):
        rollout_length = y.size(1)
        pi, mu, sigma = self.fc1(y), self.fc2(y), self.fc3(y)

        pi = pi.view(-1, rollout_length, self.n_gaussians)
        mu = mu.view(-1, rollout_length, self.n_gaussians, self.z_size)
        sigma = sigma.view(-1, rollout_length, self.n_gaussians, self.z_size)

        pi = F.softmax(pi, 2)
        sigma = torch.exp(sigma)
        return pi, mu, sigma
    def get_decoded_state_vars(self, y):
        return self.fc5(y) if self.n_state_vars>0 else None
    def forward(self, x, h):
        # Forward propagate LSTM
        y, (h, c) = self.lstm(x, h)
        pi, mu, sigma = self.get_mixture_coef(y)
        ef = self.fc4(y)
        return (pi, mu, sigma), ef, (h, c), y
    def init_hidden(self, batch_size):
        return (torch.zeros(self.n_layers, batch_size, self.n_hidden).to(config.device),
                torch.zeros(self.n_layers, batch_size, self.n_hidden).to(config.device))

def loss_pred(y_target, pi, mu, sigma, masks):
    """
        Loss associated with predicting the next latent state
    """
    y_target = y_target.unsqueeze(2)[:, :, :, :-1] # remove the "end flag" part of the target output # (batch_size, timesteps, 1, output)
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    loss = torch.exp(m.log_prob(y_target))
    #print(loss.shape, pi.unsqueeze(3).shape)
    loss = torch.sum(loss * pi.unsqueeze(3), dim=2)
    loss = -torch.log(loss) * masks[:, :, None]
    return loss.sum() / masks.sum()
def loss_errorflag(y_target, ef, masks):
    """
        Loss associated with predicting the end flag (whether the game terminated)
    """
    y_target = y_target[:, :, -1:]  # only leave the "end flag" part of the target output
    return ((y_target[:, :, -1:]-ef)**2 * masks[:, :, None]).sum() / masks.sum()
def loss_statevars(state_vars_true, state_vars_pred, masks):
    """
        Loss associated with predicting the state variables of interest (optional)
    """
    return ((state_vars_true-state_vars_pred)**2 * masks[:, :, None]).sum() / masks.sum()
# For truncated backpropagation
def detach(states): return tuple([state.detach() for state in states])