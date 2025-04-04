# ------------------- custom.py -------------------
"""
This module defines two different observation spaces ("global" vs. "local") and three different reward functions ("basic", "time_pressure", "proximity") for the Coverage Gridworld environment. 
The environment automatically calls the functions in this file to determine the observation space and compute rewards each time step.
"""
import numpy as np
import gymnasium as gym

#Global variables to control the observation and reward modes

CURRENT_OBS_MODE = "global"   # or local
CURRENT_REWARD_MODE = "basic" # or time_pressure, proximity
LAST_AGENT_POS = None  # global to track if the agent is stuck


# We'll store the last info in a global so we can do local observation without passing it around. This is kindof a workaround, but it works for this purpose.
# This gets updated each step in the reward() function, and then used in the observation() function.
LAST_INFO = None

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Observation Spaces

# 1
def global_observation_space(env: gym.Env) -> gym.spaces.Space:
    """
    flatten the entire 10x10x3 grid into a single 300-element vector
    """
    return gym.spaces.Box(low=0, high=255, shape=(300,), dtype=np.uint8)

def global_observation(grid: np.ndarray) -> np.ndarray:
    """
    Flatten the (10,10,3) grid into a vector of length (300,)
    """
    return grid.flatten()

# 2
def local_observation_space(env: gym.Env) -> gym.spaces.Space:
    """
    a 5x5 patch around the agent, plus two scalars (steps_remaining and cells_remaining). 
    That equals 77 elements (75 from 5x5x3 patch and 2 scalars).
    """
    return gym.spaces.Box(low=0, high=255, shape=(77,), dtype=np.uint8)

def local_observation(grid: np.ndarray, info: dict) -> np.ndarray:
    """
    a 5x5 grid patch around the agent's position and 2 scalars (steps_remaining and cells_remaining).
    The final shape of the array is (77,).
    """
    grid_size = 10
    agent_pos = info["agent_pos"] # This is the flattened agent position
    agent_row = agent_pos // grid_size
    agent_col = agent_pos % grid_size

    # Extract a patch of radius 2 around the row and column. Example: [row-2...row+2] and [col-2...col+2].
    patch_radius = 2
    row_min = max(0, agent_row - patch_radius)
    row_max = min(grid_size, agent_row + patch_radius + 1)
    col_min = max(0, agent_col - patch_radius)
    col_max = min(grid_size, agent_col + patch_radius + 1)

    # Extract this patch from the original 10×10×3 grid
    local_patch = grid[row_min:row_max, col_min:col_max, :]
    
    # Pad with 0's if we are near boundaries so shape stays 5x5x3
    desired_shape = (5, 5, 3)
    padded_patch = np.zeros(desired_shape, dtype=np.uint8)
    h = row_max - row_min
    w = col_max - col_min
    padded_patch[0:h, 0:w, :] = local_patch

    # Flatten the patch
    flattened_patch = padded_patch.flatten()

    # Add the scalars steps_remaining, cells_remaining
    steps_remaining = np.array([info["steps_remaining"]], dtype=np.uint8)
    cells_remaining = np.array([info["cells_remaining"]], dtype=np.uint8)

    # Combine into a single size 77 array
    local_obs = np.concatenate([flattened_patch, steps_remaining, cells_remaining])
    return local_obs

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Reward Functions

# 1
def reward_basic(info: dict) -> float:
    """
    Basic reward: 
    +1 if a new cell is covered, 
    -1 if the agent is seen (game_over)
    """
    rew = 0.0
    if info["new_cell_covered"]:
        rew += 1.0
    if info["game_over"]:
        rew -= 1.0
    return rew

# 2
def reward_time_pressure(info: dict) -> float:
    """
    Time-pressure reward:
    +1 if a new cell is covered,
    -0.01 per step,
    -5 if caught (game_over),
    -0.05 if the agent fails to move from previous step
    """
    global LAST_AGENT_POS
    rew = 0.0
    if info["new_cell_covered"]:
        rew += 1.0
    # small step penalty every time
    rew -= 0.01
    if info["game_over"]:
        rew -= 5.0

    # Check if agent's position is the same as before
    current_pos = info["agent_pos"]
    if LAST_AGENT_POS is not None:
        if current_pos == LAST_AGENT_POS:
            rew -= 0.05
    # Update position
    LAST_AGENT_POS = current_pos

    return rew

# 3
def reward_proximity(info: dict) -> float:
    """
    Proximity-based reward:
    +1 if new cell covered,
    -0.01 per step,
    -5 if caught,
    -0.1 if the agent is within a Manhattan distance of 2 from enemy
    """
    rew = 0.0
    
    # Basic coverage reward
    if info["new_cell_covered"]:
        rew += 1.0
    
    # Small time penalty each step
    rew -= 0.01
    
    # If agent is too close to an enemy, add extra penalty
    agent_pos = info["agent_pos"]
    agent_x = agent_pos % 10
    agent_y = agent_pos // 10
    
    for enemy in info["enemies"]:
        ex, ey = enemy.x, enemy.y
        dist = abs(agent_x - ex) + abs(agent_y - ey)  # Manhattan distance
        if dist < 3:  # within 2 steps
            rew -= 0.1  # too close
    
    # Caught by an enemy (game_over)
    if info["game_over"]:
        rew -= 5.0

    return rew

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Hooks for the environment to call. These are called by the enviornment so we don't need to call them directly.

def observation_space(env: gym.Env) -> gym.spaces.Space:
    """
    Returns chosen observation space.
    We'll pick based on CURRENT_OBS_MODE.
    """
    if CURRENT_OBS_MODE == "local":
        return local_observation_space(env)
    else:
        return global_observation_space(env)

def observation(grid: np.ndarray):
    """
    If we want the local approach, we rely on the global variable LAST_INFO.
    """
    global LAST_INFO

    if CURRENT_OBS_MODE == "local":
        # If info is available, do the local approach. Else, global
        if LAST_INFO is not None:
            return local_observation(grid, LAST_INFO)
        else:
            # If no info is stored yet, fallback
            return grid.flatten()
    else:
        # global flatten
        return grid.flatten()

def reward(info: dict) -> float:
    """
    Store info in LAST_INFO so 'observation' can read it if needed.
    Then we pick the function based on CURRENT_REWARD_MODE.
    """
    global LAST_INFO
    LAST_INFO = info  # store for local observation usage

    if CURRENT_REWARD_MODE == "basic":
        return reward_basic(info)
    elif CURRENT_REWARD_MODE == "time_pressure":
        return reward_time_pressure(info)
    else:
        return reward_proximity(info)