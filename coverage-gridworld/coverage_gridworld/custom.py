# ------------------- custom.py -------------------
import numpy as np
import gymnasium as gym

###############################################################################
#                             GLOBAL CONFIGS                                  #
###############################################################################
"""
Use these variables to select which observation mode and reward function 
the environment should use. The environment will call:
  - observation_space(env)
  - observation(grid)
  - reward(info)
on every step.

You can override CURRENT_OBS_MODE and CURRENT_REWARD_MODE directly in your code
or training script before creating the environment to control the behavior.
"""
CURRENT_OBS_MODE = "global"        # or "local"
CURRENT_REWARD_MODE = "basic"      # or "time_pressure", "proximity"


###############################################################################
#                               OBSERVATION SPACES                            #
###############################################################################
def global_observation_space(env: gym.Env) -> gym.spaces.Space:
    """
    Flattened 10×10×3 grid => shape (300,), each pixel 0..255 (uint8).
    """
    return gym.spaces.Box(low=0, high=255, shape=(300,), dtype=np.uint8)

def global_observation(grid: np.ndarray) -> np.ndarray:
    """
    Flatten the (10,10,3) grid into a (300,) vector.
    """
    return grid.flatten()

def local_observation_space(env: gym.Env) -> gym.spaces.Space:
    """
    We'll define a local 5×5 patch around the agent + 2 extra scalars => (5×5×3=75) + 2 = 77
    """
    return gym.spaces.Box(low=0, high=255, shape=(77,), dtype=np.uint8)

def local_observation(grid: np.ndarray, info: dict) -> np.ndarray:
    """
    Extract a 5×5 patch around the agent's position plus two scalars (steps_remaining, cells_remaining).
    """
    grid_size = 10
    # agent position in flattened form
    agent_pos = info["agent_pos"]
    agent_row = agent_pos // grid_size
    agent_col = agent_pos % grid_size

    PATCH_RADIUS = 2  # => 5×5 window
    row_min = max(0, agent_row - PATCH_RADIUS)
    row_max = min(grid_size, agent_row + PATCH_RADIUS + 1)
    col_min = max(0, agent_col - PATCH_RADIUS)
    col_max = min(grid_size, agent_col + PATCH_RADIUS + 1)

    # Slice out local patch
    local_patch = grid[row_min:row_max, col_min:col_max, :]
    
    # Pad to exactly 5×5×3 if near edges
    desired_shape = (5, 5, 3)
    padded_patch = np.zeros(desired_shape, dtype=np.uint8)
    h = row_max - row_min
    w = col_max - col_min
    padded_patch[0:h, 0:w, :] = local_patch

    # Flatten the patch
    flattened_patch = padded_patch.flatten()

    # Add scalars: steps_remaining, cells_remaining
    steps_remaining = np.array([info["steps_remaining"]], dtype=np.uint8)
    cells_remaining = np.array([info["cells_remaining"]], dtype=np.uint8)

    combined_obs = np.concatenate([flattened_patch, steps_remaining, cells_remaining])
    return combined_obs


###############################################################################
#                               REWARD FUNCTIONS                              #
###############################################################################
def reward_basic(info: dict) -> float:
    """
    Basic coverage reward: +1 if a new cell is covered, -1 if the agent is seen (game_over).
    """
    rew = 0.0
    if info["new_cell_covered"]:
        rew += 1.0
    # If the agent was seen by an enemy, game_over is True
    if info["game_over"]:
        rew -= 1.0
    return rew

def reward_time_pressure(info: dict) -> float:
    """
    +1 for each new cell covered,
    -0.01 per step,
    -5 if caught (game_over).
    """
    rew = 0.0
    if info["new_cell_covered"]:
        rew += 1.0
    # small step penalty every time
    rew -= 0.01
    if info["game_over"]:
        rew -= 5.0
    return rew

def reward_proximity(info: dict) -> float:
    """
    +1 for new cell, -0.01 step penalty, -5 if caught.
    (Optionally, you might add extra logic if you track the agent's proximity 
    to an enemy's line of sight, but that requires a custom or wrapped approach.)
    """
    rew = 0.0
    if info["new_cell_covered"]:
        rew += 1.0
    rew -= 0.01
    if info["game_over"]:
        rew -= 5.0
    return rew


###############################################################################
#             SINGLE HOOKS USED BY THE ENVIRONMENT (DO NOT RENAME!)           #
###############################################################################
def observation_space(env: gym.Env) -> gym.spaces.Space:
    """
    Called by env.py at environment init. Return your chosen observation space.
    We'll pick based on CURRENT_OBS_MODE.
    """
    if CURRENT_OBS_MODE == "local":
        return local_observation_space(env)
    else:
        return global_observation_space(env)


def observation(grid: np.ndarray):
    """
    Called by env.py each time it needs to return an observation.

    Note: We do NOT have direct access to `info` here in this function signature.
    Because the local approach might need `info`, we handle that in a wrapper approach 
    OR we store the last info globally. However, by default the environment only 
    passes the grid.

    In this demonstration, we simply return the global flattened observation 
    if we rely on env.py's direct call. For a truly local approach that depends on info, 
    a wrapper is recommended. 
    """
    if CURRENT_OBS_MODE == "local":
        # We can't do it properly here because we don't have `info`.
        # We'll default to returning global flatten. 
        # (Alternatively, use a custom ObservationWrapper instead.)
        return grid.flatten()
    else:
        # global flatten
        return grid.flatten()


def reward(info: dict) -> float:
    """
    Called by env.py each time it needs to compute reward.
    We pick the function based on CURRENT_REWARD_MODE.
    """
    if CURRENT_REWARD_MODE == "basic":
        return reward_basic(info)
    elif CURRENT_REWARD_MODE == "time_pressure":
        return reward_time_pressure(info)
    else:  # "proximity"
        return reward_proximity(info)
