from stable_baselines3 import PPO
import gymnasium as gym
import coverage_gridworld
import custom

custom.CURRENT_OBS_MODE = "global"
custom.CURRENT_REWARD_MODE = "basic"

env = gym.make("safe", render_mode="human")
model = PPO.load("my_coverage_agent_global_basic")

obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    if truncated:
        done = True
