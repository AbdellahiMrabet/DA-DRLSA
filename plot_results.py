# plot_results.py - generate training curves and policy evaluation charts
import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Set font styling for publication quality
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.autolayout': True
})

def plot_training(log_path="ckpt/train_log.json", out_path="results/fig_training_curves.png"):
    if not os.path.exists(log_path):
        print(f"Skipping training curves: {log_path} not found.")
        return

    with open(log_path, "r") as f:
        log = json.load(f)

    episodes = [row["ep"] for row in log]
    rewards = [row["reward"] for row in log]
    miss_ratios = [row["miss"] * 100 for row in log]
    hard_miss = [row["hard"] * 100 for row in log]

    # Smooth curves with a moving average window
    def smooth(vals, w=10):
        if len(vals) < w:
            return vals
        return np.convolve(vals, np.ones(w)/w, mode='valid')

    smoothed_eps = episodes[len(episodes) - len(smooth(rewards)):]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Reward Curve
    ax1.plot(episodes, rewards, color='lightblue', alpha=0.4, label='Raw Episode Reward')
    ax1.plot(smoothed_eps, smooth(rewards), color='blue', linewidth=2, label='Smoothed (10-ep MA)')
    ax1.set_title('Training Episode Reward')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right')

    # Miss Ratio Curve
    ax2.plot(episodes, miss_ratios, color='lightcoral', alpha=0.4, label='Raw All Miss Ratio (%)')
    ax2.plot(smoothed_eps, smooth(miss_ratios), color='crimson', linewidth=2, label='All Tasks Miss Ratio')
    ax2.plot(smoothed_eps, smooth(hard_miss), color='darkred', linestyle='--', linewidth=2, label='Hard Tasks Miss Ratio')
    ax2.set_title('Deadline Miss Ratios During Training')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Miss Ratio (%)')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right')

    plt.savefig(out_path, dpi=300)
    print(f"Saved training curves to {out_path}")
    plt.close()


def plot_evaluation(results_path="results.json", out_path="results/fig_policy_comparison.png"):
    if not os.path.exists(results_path):
        print(f"Skipping evaluation comparison: {results_path} not found. Run evaluate.py first.")
        return

    with open(results_path, "r") as f:
        results = json.load(f)

    policies = list(results.keys())
    x = np.arange(len(policies))
    width = 0.25

    all_miss = [results[p]["miss_all"][0] * 100 for p in policies]
    hard_miss = [results[p]["miss_hard"][0] * 100 for p in policies]
    p99_latency = [results[p]["p99_hard_s"][0] for p in policies]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Deadline Miss Ratios Comparison
    rects1 = ax1.bar(x - width/2, all_miss, width, label='All Tasks Miss (%)', color='skyblue')
    rects2 = ax1.bar(x + width/2, hard_miss, width, label='Hard Tasks Miss (%)', color='salmon')
    ax1.set_title('Deadline Miss Ratio by Policy')
    ax1.set_ylabel('Miss Ratio (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(policies, rotation=15)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.legend()

    # P99 Hard Latency Comparison
    ax2.bar(x, p99_latency, width*1.5, color='mediumpurple')
    ax2.set_title('P99 Hard Task Execution Latency')
    ax2.set_ylabel('Latency (s)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(policies, rotation=15)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    plt.savefig(out_path, dpi=300)
    print(f"Saved policy comparison charts to {out_path}")
    plt.close()


if __name__ == "__main__":
    plot_training()
    plot_evaluation()
