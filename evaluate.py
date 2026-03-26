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


# run a purely random agent for num_episodes, capping each episode at max_steps
def evaluate_random(num_episodes=100, max_steps=2000):
    env = Game2048()
    scores = []
    max_tiles = []

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            action = np.random.randint(4)
            state, _, done, info = env.step(action)
            steps += 1
        scores.append(info['score'])
        max_tiles.append(int(env.board.max()))

    return scores, max_tiles


# apply a uniform moving average to smooth a 1d data array
def smooth(data, window=50):
    if len(data) < window:
        return np.array(data)
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


# save a polished bar chart comparing average game scores, with exact values above each bar
def plot_avg_score(eval_results, keys, labels, colors):
    avg_scores = [np.mean(eval_results[k]['scores']) for k in keys]
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(range(len(labels)), avg_scores, color=colors,
                  width=0.55, edgecolor='white', linewidth=1.2)
    top = max(avg_scores)
    for bar, val in zip(bars, avg_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, val + top * 0.013,
                f'{val:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('average score over evaluation episodes', fontsize=12)
    ax.set_title('average game score per experiment group', fontsize=14,
                 fontweight='bold', pad=12)
    ax.set_ylim(0, top * 1.18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=2.0)
    fig.savefig('plots/avg_score.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("saved plots/avg_score.png")


# save a polished bar chart of mean log2 max tile per group, with exact values above each bar
def plot_avg_max_tile(eval_results, keys, labels, colors):
    avg_log2 = [
        np.mean([np.log2(t) if t > 0 else 0 for t in eval_results[k]['max_tiles']])
        for k in keys
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(range(len(labels)), avg_log2, color=colors,
                  width=0.55, edgecolor='white', linewidth=1.2)
    top = max(avg_log2)
    for bar, val in zip(bars, avg_log2):
        ax.text(bar.get_x() + bar.get_width() / 2, val + top * 0.013,
                f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('mean max tile, log2 scale', fontsize=12)
    ax.set_title('average max tile reached per experiment group', fontsize=14,
                 fontweight='bold', pad=12)
    ax.set_ylim(0, top * 1.18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=2.0)
    fig.savefig('plots/avg_max_tile_log2.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("saved plots/avg_max_tile_log2.png")


# save a polished reward curve chart using distinct colors and line styles for each group
def plot_reward_curves(training_results, keys, labels, colors):
    line_styles = ['-', '--', '-.', ':']
    fig, ax = plt.subplots(figsize=(13, 7))
    training_keys = [k for k in keys if k != 'random']
    for i, key in enumerate(training_keys):
        rewards = training_results[key]['rewards']
        smoothed = smooth(rewards, window=50)
        x = np.arange(len(smoothed)) + 25
        idx = keys.index(key)
        ax.plot(x, smoothed, label=labels[idx], color=colors[idx],
                linestyle=line_styles[i], linewidth=2.0)
    ax.set_xlabel('episode', fontsize=12)
    ax.set_ylabel('total reward, smoothed with window 50', fontsize=12)
    ax.set_title('training reward curves per experiment group', fontsize=14,
                 fontweight='bold', pad=12)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=2.0)
    fig.savefig('plots/reward_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("saved plots/reward_curves.png")


# load checkpoints, evaluate all groups and the random baseline, then generate all plots
def run_evaluation(num_eval_episodes=100):
    with open('results/training_results.pkl', 'rb') as f:
        training_results = pickle.load(f)

    os.makedirs('plots', exist_ok=True)

    keys = ['group_1', 'group_2', 'group_3', 'group_4', 'random']
    labels = [
        'group 1, score only',
        'group 2, game over penalty',
        'group 3, invalid move penalty',
        'group 4, both penalties',
        'random baseline'
    ]
    colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson', 'gray']

    eval_results = {}

    for group_id in range(1, 5):
        key = f'group_{group_id}'
        agent = DQNAgent()
        agent.load(f'checkpoints/{key}.pth')
        scores, max_tiles = evaluate_agent(agent, num_eval_episodes)
        eval_results[key] = {'scores': scores, 'max_tiles': max_tiles}
        print(f"group {group_id}: avg score {np.mean(scores):.1f}, "
              f"avg max tile {np.mean(max_tiles):.1f}")

    scores, max_tiles = evaluate_random(num_eval_episodes)
    eval_results['random'] = {'scores': scores, 'max_tiles': max_tiles}
    print(f"random baseline: avg score {np.mean(scores):.1f}, "
          f"avg max tile {np.mean(max_tiles):.1f}")

    plot_avg_score(eval_results, keys, labels, colors)
    plot_avg_max_tile(eval_results, keys, labels, colors)
    plot_reward_curves(training_results, keys, labels, colors)

    with open('results/eval_results.pkl', 'wb') as f:
        pickle.dump(eval_results, f)

    print("\nevaluation complete, plots saved to plots/")
    return eval_results


if __name__ == '__main__':
    run_evaluation()
