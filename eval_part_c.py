import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from game_env import Game2048
from dqn_agent import DQNAgent


# run a trained agent with epsilon 0.05 for num_episodes, capping each episode at max_steps
def evaluate_agent(agent, num_episodes=100, max_steps=2000):
    env = Game2048()
    scores = []
    max_tiles = []
    saved_epsilon = agent.epsilon
    agent.epsilon = 0.05

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            action = agent.select_action(state)
            state, _, done, info = env.step(action)
            steps += 1
        scores.append(info['score'])
        max_tiles.append(int(env.board.max()))

    agent.epsilon = saved_epsilon
    return scores, max_tiles


# apply a uniform moving average to smooth a 1d data array
def smooth(data, window=50):
    if len(data) < window:
        return np.array(data)
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


# save two line plots showing how each penalty type's magnitude affects average evaluation score
def plot_parameter_study(eval_results, out_path):
    baseline_avg = np.mean(eval_results['ps_baseline']['scores'])

    go_magnitudes = [50, 100, 200]
    go_avg_scores = [np.mean(eval_results[k]['scores']) for k in ['ps1', 'ps2', 'ps3']]

    inv_magnitudes = [2, 5, 10]
    inv_avg_scores = [np.mean(eval_results[k]['scores']) for k in ['ps4', 'ps5', 'ps6']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Effect of Penalty Magnitude on Agent Performance (Part C)',
                 fontsize=14, fontweight='bold')

    ax1.plot(go_magnitudes, go_avg_scores, 'o-', color='steelblue', linewidth=2, markersize=8)
    ax1.axhline(y=baseline_avg, color='gray', linestyle='--', label='no penalty', alpha=0.7)
    ax1.set_xlabel('game over penalty magnitude', fontsize=12)
    ax1.set_ylabel('average evaluation score', fontsize=12)
    ax1.set_title('game over penalty', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax1.set_axisbelow(True)

    ax2.plot(inv_magnitudes, inv_avg_scores, 'o-', color='crimson', linewidth=2, markersize=8)
    ax2.axhline(y=baseline_avg, color='gray', linestyle='--', label='no penalty', alpha=0.7)
    ax2.set_xlabel('invalid move penalty magnitude', fontsize=12)
    ax2.set_ylabel('average evaluation score', fontsize=12)
    ax2.set_title('invalid move penalty', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax2.set_axisbelow(True)

    fig.tight_layout(pad=2.0)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {out_path}")


# save a bar chart comparing average game scores for all parameter study groups
def plot_avg_score(eval_results, keys, labels, colors, out_path):
    avg_scores = [np.mean(eval_results[k]['scores']) for k in keys]
    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(range(len(labels)), avg_scores, color=colors,
                  width=0.6, edgecolor='white', linewidth=1.2)
    top = max(avg_scores)
    for bar, val in zip(bars, avg_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, val + top * 0.013,
                f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('average score over evaluation episodes', fontsize=11)
    ax.set_title('average game score per parameter study group (part c)', fontsize=14,
                 fontweight='bold', pad=12)
    ax.set_ylim(0, top * 1.18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=2.0)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {out_path}")


# save a reward curve chart for all parameter study groups
def plot_reward_curves(training_results, keys, labels, colors, out_path):
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
    fig, ax = plt.subplots(figsize=(13, 6))
    for i, key in enumerate(keys):
        rewards = training_results[key]['rewards']
        smoothed = smooth(rewards, window=50)
        x = np.arange(len(smoothed)) + 25
        ax.plot(x, smoothed, label=labels[i], color=colors[i],
                linestyle=line_styles[i], linewidth=2.0)
    ax.set_xlabel('episode', fontsize=12)
    ax.set_ylabel('total reward, smoothed with window 50', fontsize=12)
    ax.set_title('training reward curves (part c)', fontsize=14,
                 fontweight='bold', pad=12)
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=2.0)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {out_path}")


# load checkpoints, evaluate all parameter study groups, and generate all part c plots
def run_evaluation_c(num_eval_episodes=100):
    with open('results/part_c_training_results.pkl', 'rb') as f:
        training_results = pickle.load(f)

    os.makedirs('plots', exist_ok=True)

    keys = ['ps_baseline', 'ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'ps6']
    labels = [
        'PS baseline',
        'PS1 GO=-50',
        'PS2 GO=-100',
        'PS3 GO=-200',
        'PS4 INV=-2',
        'PS5 INV=-5',
        'PS6 INV=-10',
    ]
    colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948', '#b07aa1']

    eval_results = {}

    for key, label in zip(keys, labels):
        agent = DQNAgent(use_double_dqn=False)
        agent.load(f'checkpoints/group_{key}.pth')
        scores, max_tiles = evaluate_agent(agent, num_eval_episodes)
        eval_results[key] = {'scores': scores, 'max_tiles': max_tiles}
        print(f"{label}: avg score {np.mean(scores):.1f}, "
              f"avg max tile {np.mean(max_tiles):.1f}")

    plot_parameter_study(eval_results, 'plots/part_c_parameter_study.png')
    plot_avg_score(eval_results, keys, labels, colors, 'plots/part_c_avg_score.png')
    plot_reward_curves(training_results, keys, labels, colors, 'plots/part_c_reward_curves.png')

    with open('results/part_c_eval_results.pkl', 'wb') as f:
        pickle.dump(eval_results, f)

    print("\npart c evaluation complete, plots saved to plots/")
    return eval_results


if __name__ == '__main__':
    run_evaluation_c()
