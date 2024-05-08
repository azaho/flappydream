import flappy_bird_gymnasium_fork
import gymnasium
import numpy as np
import time
import pygame
from flappy_bird_gymnasium.tests.dueling import DuelingDQN
from pathlib import Path
import config

def extract_rollouts(N_rollouts, player_type, filename=None, dqn_sudden_death_p=0.01, random_policy_flap_p=0.075, verbose=False, save_pixel_data=False, extraction_id=0):
    """
    Saves rollouts
    :param N_rollouts:
    :param player_type: can be random or dqn (dqn turns into random with 1% probability every timestep to avoid runs which are too long)
    :param save_pixel_data: (optional) alongside the rollouts file, save all the 64x64x3 pixel observations for every timestep.
    """
    if filename is None:
        filename = f"data/rollouts_{N_rollouts}_{player_type}_{extraction_id}.npz"
        Path("data/").mkdir(parents=True, exist_ok=True)
    env = gymnasium.make("FlappyBirdFork-v0", render_mode="rgb_array")
    rollout_sum_reward_store, rollout_game_score_store, observation_store = [], [], []
    action_store, reward_store, end_flag_store = [], [], []
    pixel_data_store = []
    if player_type == "dqn":
        q_model = DuelingDQN(env.action_space.n)
        q_model.build((None, 12))
        q_model.load_weights("flappy_bird_gymnasium_fork/dqn_model.h5")
    if verbose:
        print(f"===Collecting {N_rollouts} rollouts of {player_type} policy")
    for rollout_i in range(N_rollouts):
        rollout_sum_reward = 0

        obs, _ = env.reset()
        observation_store.append(obs)
        reward_store.append(0)
        end_flag_store.append(0)
        if save_pixel_data: pixel_data_store.append(env.get_pixel_data())

        sudden_death_flag = False
        while True:
            # Next action:
            if player_type == "random" or sudden_death_flag:
                action = 1 if np.random.rand() < random_policy_flap_p else 0
            elif player_type == "dqn":
                state = np.expand_dims(obs[180:], axis=0)
                action = q_model.get_action(state)
            action_store.append(action)
            # sudden switch of expert to random (to avoid very long rollouts)
            if np.random.rand() < dqn_sudden_death_p: sudden_death_flag = True
            # Processing:
            obs, reward, terminated, _, info = env.step(action)
            observation_store.append(obs)
            reward_store.append(reward)
            end_flag_store.append(int(terminated))
            if save_pixel_data: pixel_data_store.append(env.get_pixel_data())
            rollout_sum_reward += reward
            if terminated:
                action_store.append(0)
                rollout_game_score_store.append(env.get_game_score())
                rollout_sum_reward_store.append(rollout_sum_reward)
                break
        if verbose and rollout_i % 10 == 0: print(rollout_i)
    env.close()
    observation_store = np.array(observation_store)
    action_store = np.array(action_store)
    reward_store = np.array(reward_store)
    end_flag_store = np.array(end_flag_store)
    rollout_game_score_store = np.array(rollout_game_score_store)
    rollout_sum_reward_store = np.array(rollout_sum_reward_store)
    pixel_data_store = np.array(pixel_data_store)
    np.savez_compressed(filename,
                        observation_store=observation_store, action_store=action_store,
                        reward_store=reward_store, end_flag_store=end_flag_store,
                        rollout_game_score_store=rollout_game_score_store,
                        rollout_sum_reward_store=rollout_sum_reward_store)
    if save_pixel_data:
        np.savez_compressed(filename[:-4]+"_pixeldata.npz",
                            pixel_data_store=pixel_data_store)
    if verbose:
        print(f"\n===SAVED.\nFrom {len(rollout_game_score_store)} runs, {np.sum(rollout_game_score_store > 0)} achieved >0 score")


if __name__ == "__main__":
    # for extraction_id in range(1, 4):
    #     extract_rollouts(config.rollouts_dqn_n, "dqn", verbose=True, save_pixel_data=True, extraction_id=extraction_id)
    extract_rollouts(config.rollouts_dqn_n, "dqn", verbose=True, save_pixel_data=False, extraction_id=5)
    extract_rollouts(config.rollouts_random_n, "random", verbose=True, save_pixel_data=False, extraction_id=5)