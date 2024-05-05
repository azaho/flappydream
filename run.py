import flappy_bird_gymnasium_fork
import gymnasium
import numpy as np
import time
import pygame
import vae
import torch
from flappy_bird_gymnasium.tests.dueling import DuelingDQN
env = gymnasium.make("FlappyBirdFork-v0", render_mode="rgb_array")

latent_dim = 12

class MyCustomWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def render(self):
        arr = super().render()
        #if arr is not None: print(arr.shape)
        return arr
env = MyCustomWrapper(env)
env = gymnasium.wrappers.HumanRendering(env)
#env = gymnasium.wrappers.RecordVideo(env, video_folder="videos/")


q_model = DuelingDQN(env.action_space.n)
q_model.build((None, 12))
#q_model.load_weights("flappy_bird_gymnasium_fork/assets/model/model.h5")

player_type = "random"
player_type = "human"
player_type = "dqn"

device = torch.device("cpu")
model = vae.VAE(device, latent_dim=latent_dim).to(device)
model.load_state_dict(torch.load(f"vae_model_{latent_dim}dimlatent.pt", map_location=torch.device('cpu')))
def vae_function(x):
    x_hat, mean, log_var = model(torch.tensor(x, dtype=torch.float32))
    x_hat = x_hat.cpu().detach().numpy()
    return x_hat
env.set_show_vae(vae_function)

while True:

    player_type = "dqn"

    observation_store = []
    reward_store = []
    terminated = []

    obs, _ = env.reset()
    while True:
        # Next action:
        # (feed the observation to your agent here)
        if player_type == "random":
            action = env.action_space.sample()
            if np.random.rand()>0.925: action = 1
            else: action = 0
        elif player_type=="human":
            action = 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                if event.type == pygame.KEYDOWN and (
                        event.key == pygame.K_SPACE or event.key == pygame.K_UP
                ):
                    action = 1
        elif player_type=="dqn":
            state = obs[180:]
            state = np.expand_dims(state, axis=0)
            #print(obs.shape)
            #print(state.shape)
            action = q_model.get_action(state)
        #print(action)

        if np.random.rand()<0.01:
            #player_type = "random"
            print("switch")

        # Processing:
        obs, reward, terminated, _, info = env.step(action)
        #print(obs.shape)
        #time.sleep(.05)
        # Checking if the player is still alive
        if terminated:
            break

env.close()
