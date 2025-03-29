# ------------------- train.py -------------------
import gymnasium as gym
import coverage_gridworld  # Needed so the gym environments are registered
from stable_baselines3 import PPO
import custom  # This is our custom.py file

"""
Usage:
    1. pip install -e coverage-gridworld   # ensures the environment is available
    2. pip install stable-baselines3 gymnasium matplotlib
    3. python train.py
"""

if __name__ == "__main__":
    # Choose an observation mode and a reward mode:
    # (e.g., "global" obs + "basic" reward)
    custom.CURRENT_OBS_MODE = "global"
    custom.CURRENT_REWARD_MODE = "basic"

    # Make the environment (example: 'just_go', 'sneaky_enemies', or 'standard' for random)
    env = gym.make("sneaky_enemies", render_mode=None, activate_game_status=False)

    # Create and train PPO
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    # Save the trained agent
    model.save("my_coverage_agent_global_basic")

    # ---------------
    # Example: train with a different combo (global + time_pressure)
    custom.CURRENT_OBS_MODE = "global"
    custom.CURRENT_REWARD_MODE = "time_pressure"

    env2 = gym.make("sneaky_enemies", render_mode=None, activate_game_status=False)
    model2 = PPO("MlpPolicy", env2, verbose=1)
    model2.learn(total_timesteps=10000)
    model2.save("my_coverage_agent_global_time")

    # ---------------
    # Example: train with a different combo (global + proximity)
    custom.CURRENT_OBS_MODE = "global"
    custom.CURRENT_REWARD_MODE = "proximity"

    env3 = gym.make("sneaky_enemies", render_mode=None, activate_game_status=False)
    model3 = PPO("MlpPolicy", env3, verbose=1)
    model3.learn(total_timesteps=10000)
    model3.save("my_coverage_agent_global_proximity")

    # Etc. 
    # If you want "local" observation, set custom.CURRENT_OBS_MODE = "local"
    # but be aware that by default, 'observation(grid)' lacks 'info', so you'd 
    # typically create a wrapper or store last info in the env to do it properly.
