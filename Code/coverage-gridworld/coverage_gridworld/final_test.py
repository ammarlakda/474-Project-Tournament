import gymnasium as gym
import coverage_gridworld
from stable_baselines3 import PPO
import custom

custom.CURRENT_OBS_MODE = "local"
custom.CURRENT_REWARD_MODE = "proximity"

# Path to model
model_path = "final_model_local_proximity.zip"

env = gym.make("sneaky_enemies", render_mode="human", activate_game_status=True)
model = PPO.load(model_path, env=env, verbose=1)

# Evaluate for multiple episodes
NUM_EPISODES = 3
for episode in range(NUM_EPISODES):
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    step_count = 0

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        if truncated:
            done = True

    print(f"Episode {episode+1} finished with total reward: {total_reward}, steps: {step_count}")

env.close()
