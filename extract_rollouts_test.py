from extract_rollouts import *
import config

if __name__ == "__main__":
    extract_rollouts(config.rollouts_dqn_n, "dqn", verbose=True, save_pixel_data=False, extraction_id=100)
    extract_rollouts(config.rollouts_dqn_n, "random", verbose=True, save_pixel_data=False, extraction_id=100)