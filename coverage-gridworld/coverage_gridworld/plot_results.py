import os
import pandas as pd
import matplotlib.pyplot as plt

# Define your combos
COMBOS = [
    ("global", "basic"),
    ("global", "time_pressure"),
    ("global", "proximity"),
    ("local", "basic"),
    ("local", "time_pressure"),
    ("local", "proximity")
]

def load_rewards(obs_mode, rew_mode):
    """
    Reads 'monitor.csv' from ./logs/{obs_mode}_{rew_mode}/monitor.csv
    and returns a pandas Series of episode rewards.
    
    Returns None if the file is missing or empty.
    """
    log_dir = f"./logs/{obs_mode}_{rew_mode}"
    monitor_file = os.path.join(log_dir, "monitor.csv")
    
    if not os.path.exists(monitor_file):
        print(f"No monitor.csv found in {log_dir} – skipping.")
        return None
    
    # Stable Baselines Monitor logs have a commented first line, so skiprows=1
    df = pd.read_csv(monitor_file, skiprows=1)
    if len(df) == 0:
        print(f"Empty monitor.csv in {log_dir} – skipping.")
        return None
    
    # 'r' is typically the column for per-episode reward
    return df["r"]


def plot_rolling_average_line(combos, window=10):
    """
    1) Rolling Average Line Plot:
       Plots all combos on ONE figure, each with a separate line for reward over episodes.
       We use a rolling average of the reward to smooth the curve.
    """
    plt.figure()
    for obs_mode, rew_mode in combos:
        rewards = load_rewards(obs_mode, rew_mode)
        if rewards is None:
            continue
        
        rolling = rewards.rolling(window).mean()  # rolling average
        label = f"{obs_mode}-{rew_mode}"
        plt.plot(rolling, label=label)
    
    plt.title(f"Rolling-Average Episode Reward (window={window})")
    plt.xlabel("Episode Index")
    plt.ylabel("Rolling Avg Reward")
    plt.legend()
    plt.show()

def plot_overlapping_histograms(combos, bins=20):
    """
    2) Overlapping Histograms:
       Plots a single figure with multiple reward distributions (histograms) overlapped.
    """
    plt.figure()
    for obs_mode, rew_mode in combos:
        rewards = load_rewards(obs_mode, rew_mode)
        if rewards is None:
            continue
        
        label = f"{obs_mode}-{rew_mode}"
        # alpha=0.5 => semi-transparent so they overlap better
        plt.hist(rewards, bins=bins, alpha=0.5, label=label)
    
    plt.title("Overlapping Histograms of Episode Rewards")
    plt.xlabel("Episode Reward")
    plt.ylabel("Count")
    plt.legend()
    plt.show()

def plot_box_and_whisker(combos):
    """
    3) Box-and-Whisker Plot:
       Places a box for each combo side-by-side on a single figure.
    """
    all_data = []
    labels = []
    
    for obs_mode, rew_mode in combos:
        rewards = load_rewards(obs_mode, rew_mode)
        if rewards is not None:
            all_data.append(rewards)
            labels.append(f"{obs_mode}\n{rew_mode}")
    
    if len(all_data) == 0:
        print("No data to plot for boxplot.")
        return
    
    plt.figure()
    # showmeans=True => a small symbol (triangle) is used to indicate the mean
    plt.boxplot(all_data, labels=labels, showmeans=True)
    plt.title("Episode Rewards Box-and-Whisker")
    plt.ylabel("Reward")
    plt.show()

if __name__ == "__main__":
    # 1) Rolling average line plot
    plot_rolling_average_line(COMBOS, window=10)

    # 2) Overlapping histograms
    plot_overlapping_histograms(COMBOS, bins=20)

    # 3) Box-and-whisker
    plot_box_and_whisker(COMBOS)
