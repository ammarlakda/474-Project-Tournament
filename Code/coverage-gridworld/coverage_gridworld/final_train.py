import os
import gymnasium as gym
import coverage_gridworld
import custom
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

if __name__ == "__main__":
    # 1) This is the best observation + reward combo from previous experiments
    custom.CURRENT_OBS_MODE = "local"
    custom.CURRENT_REWARD_MODE = "proximity"

    # STAGE 1: Train on "safe"
    env1 = gym.make("safe", render_mode=None, activate_game_status=False)
    log_dir1 = "./logs/stage1_safe"
    os.makedirs(log_dir1, exist_ok=True)
    env1 = Monitor(env1, log_dir1)

    model = PPO("MlpPolicy", env1, verbose=1)
    model.learn(total_timesteps=50_000)  # 50k steps on "safe"
    model.save("stage1_local_proximity_safe.zip")
    env1.close()

    # STAGE 2: Load the model and train on "chokepoint" (harder)
    env2 = gym.make("chokepoint", render_mode=None, activate_game_status=False)
    log_dir2 = "./logs/stage2_chokepoint"
    os.makedirs(log_dir2, exist_ok=True)
    env2 = Monitor(env2, log_dir2)

    model = PPO.load("stage1_local_proximity_safe.zip", env=env2, verbose=1)
    model.learn(total_timesteps=70_000)  # 70k steps on "chokepoint"
    model.save("stage2_local_proximity_chokepoint.zip")
    env2.close()


    # STAGE 3: Load the model again, now move to "sneaky_enemies" (very hard)
    env3 = gym.make("sneaky_enemies", render_mode=None, activate_game_status=False)
    log_dir3 = "./logs/stage3_sneaky_enemies"
    os.makedirs(log_dir3, exist_ok=True)
    env3 = Monitor(env3, log_dir3)

    model = PPO.load("stage2_local_proximity_chokepoint.zip", env=env3, verbose=1)
    model.learn(total_timesteps=100_000)  # 100k steps
    model.save("final_model_local_proximity.zip")
    env3.close()

    print("Training complete! Model saved as final_model_local_proximity.zip")
