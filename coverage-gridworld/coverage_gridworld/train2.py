# train.py
import gymnasium as gym
from stable_baselines3 import PPO
import coverage_gridworld
import custom

from stable_baselines3.common.monitor import Monitor
import os

if __name__ == "__main__":
    observation_modes = ["global", "local"]
    reward_modes = ["basic", "time_pressure", "proximity"]
    
    total_timesteps = 30000  # or more, e.g. 100000

    for obs_mode in observation_modes:
        for rew_mode in reward_modes:
            # 1. Set your custom.py config
            custom.CURRENT_OBS_MODE = obs_mode
            custom.CURRENT_REWARD_MODE = rew_mode

            # 2. Create log directory for this combo
            log_dir = f"./logs/{obs_mode}_{rew_mode}"
            os.makedirs(log_dir, exist_ok=True)

            # 3. Make the environment & wrap it with Monitor
            env = gym.make("sneaky_enemies", render_mode=None, activate_game_status=False)
            env = Monitor(env, log_dir)

            # 4. Train PPO
            model = PPO("MlpPolicy", env, verbose=1)
            model.learn(total_timesteps=total_timesteps)

            # 5. Save the model
            model.save(f"{obs_mode}_{rew_mode}_model.zip")

            env.close()
