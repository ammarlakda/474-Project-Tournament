import pandas as pd
import matplotlib.pyplot as plt
import os

def load_rewards(log_dir):
    """
    Reads monitor.csv files from the logs for episode rewards.
    """
    monitor_file = os.path.join(log_dir, "monitor.csv")
    if not os.path.exists(monitor_file):
        print(f"Warning: no monitor.csv in {log_dir} - skipping.")
        return None
    df = pd.read_csv(monitor_file, skiprows=1)
    rewards = df["r"].values
    return rewards

if __name__ == "__main__":
    # Define the combos
    combos = [
        ("global", "basic"),
        ("global", "time_pressure"),
        ("global", "proximity"),
        ("local", "basic"),
        ("local", "time_pressure"),
        ("local", "proximity"),
    ]

    # Collect data from each combo
    all_data = []
    all_labels = []
    for (obs_mode, rew_mode) in combos:
        log_dir = f"./logs/{obs_mode}_{rew_mode}"
        rewards = load_rewards(log_dir)
        if rewards is not None and len(rewards) > 0:
            all_data.append(rewards)
            all_labels.append(f"{obs_mode}\n{rew_mode}")
        else:
            pass

    # Create plots
    if len(all_data) > 0:
        # 1) Boxplot
        plt.figure()
        plt.boxplot(all_data, labels=all_labels, showmeans=True)
        plt.title("Episode Rewards Distribution by (Obs, Reward) Combo")
        plt.xlabel("Observation / Reward Combos")
        plt.ylabel("Episode Return")
        plt.show()
        
        # 2) Bar plot
        mean_rewards = [rewards.mean() for rewards in all_data]
        plt.figure()
        plt.bar(range(len(mean_rewards)), mean_rewards, tick_label=all_labels)
        plt.title("Average Episode Reward by (Obs, Reward) Combo")
        plt.xlabel("Observation / Reward Combos")
        plt.ylabel("Mean Episode Return")
        plt.show()

        # 3) Overlapping histogram
        plt.figure()
        for data, label in zip(all_data, all_labels):
            # alpha=0.5 => semi-transparent so multiple histograms can overlap
            plt.hist(data, bins=20, alpha=0.5, label=label)

        plt.title("Overlapping Histograms of Episode Rewards")
        plt.xlabel("Episode Reward")
        plt.ylabel("Count")
        plt.legend()
        plt.show()

        # 4) Rolling average line plot
        # Create subplots
        fig, axs = plt.subplots(nrows=len(all_data), ncols=1, figsize=(8, 3 * len(all_data)))
        if len(all_data) == 1:
            axs = [axs]

        window_size = 10
        for i, (rewards, label) in enumerate(zip(all_data, all_labels)):
            # Compute rolling average
            series = pd.Series(rewards)
            rolling_avg = series.rolling(window_size).mean()
            # Plot each combo in its own subplot
            ax = axs[i]
            ax.plot(rolling_avg, alpha=0.8)
            ax.set_title(f"Rolling Avg Rewards ({label}) [window={window_size}]")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Reward")
        plt.tight_layout(h_pad=2.0)
        plt.show()
    else:
        print("No data found to plot.")
