import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
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


# save a bar chart comparing average game scores, with exact values above each bar
def plot_avg_score(eval_results, keys, labels, colors, out_path):
    avg_scores = [np.mean(eval_results[k]['scores']) for k in keys]
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(labels)), avg_scores, color=colors,
                  width=0.6, edgecolor='white', linewidth=1.2)
    top = max(avg_scores)
    for bar, val in zip(bars, avg_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, val + top * 0.013,
                f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('average score over evaluation episodes', fontsize=11)
    ax.set_title('average game score per experiment group (part b)', fontsize=14,
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


# save a bar chart of mean log2 max tile per group, with exact values above each bar
def plot_avg_max_tile(eval_results, keys, labels, colors, out_path):
    avg_log2 = [
        np.mean([np.log2(t) if t > 0 else 0 for t in eval_results[k]['max_tiles']])
        for k in keys
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(labels)), avg_log2, color=colors,
                  width=0.6, edgecolor='white', linewidth=1.2)
    top = max(avg_log2)
    for bar, val in zip(bars, avg_log2):
        ax.text(bar.get_x() + bar.get_width() / 2, val + top * 0.013,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('mean max tile, log2 scale', fontsize=11)
    ax.set_title('average max tile reached per experiment group (part b)', fontsize=14,
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


# save a reward curve chart with distinct colors and line styles for each trained group
def plot_reward_curves(training_results, keys, labels, colors, out_path):
    line_styles = ['--', '-.', ':', '-', '--']
    fig, ax = plt.subplots(figsize=(12, 6))
    trained_keys = [k for k in keys if k != 'random']
    for i, key in enumerate(trained_keys):
        rewards = training_results[key]['rewards']
        smoothed = smooth(rewards, window=50)
        x = np.arange(len(smoothed)) + 25
        idx = keys.index(key)
        ax.plot(x, smoothed, label=labels[idx], color=colors[idx],
                linestyle=line_styles[i], linewidth=2.0)
    ax.set_xlabel('episode', fontsize=12)
    ax.set_ylabel('total reward, smoothed with window 50', fontsize=12)
    ax.set_title('training reward curves (part b)', fontsize=14,
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


# save a 1x4 grid of stacked area charts showing action proportions during training
def plot_action_distribution(training_results, group_keys, group_labels, out_path):
    action_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    action_names = ['up', 'down', 'left', 'right']
    window = 200

    fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharex=True, sharey=True)

    for i, (key, label) in enumerate(zip(group_keys, group_labels)):
        ax = axes[i]
        raw_counts = training_results[key].get('action_counts', [])
        if len(raw_counts) == 0:
            ax.set_title(label, fontsize=9, fontweight='bold')
            continue

        counts = np.array(raw_counts, dtype=np.float64)
        n = len(counts)

        # compute rolling proportions using a convolution based moving sum
        if n >= window:
            rolling = np.zeros((n - window + 1, 4))
            for j in range(4):
                rolling[:, j] = np.convolve(counts[:, j], np.ones(window), mode='valid')
            totals = rolling.sum(axis=1, keepdims=True)
            totals[totals == 0] = 1
            proportions = rolling / totals
            x = np.arange(window // 2, window // 2 + len(proportions))
        else:
            totals = counts.sum(axis=1, keepdims=True)
            totals[totals == 0] = 1
            proportions = counts / totals
            x = np.arange(n)

        ax.stackplot(x, proportions.T, labels=action_names, colors=action_colors, alpha=0.8)
        ax.set_title(label, fontsize=9, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_xlabel('episode', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylabel('action proportion', fontsize=10)

    # shared legend from the first subplot
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='lower center', ncol=4, fontsize=10, framealpha=0.9)

    fig.suptitle('action distribution during training, rolling window 200 (part b)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {out_path}")


# save a 1x4 grid of bar charts showing q values for a fixed board state across groups 5 through 8
def plot_qvalue_heatmap(group_keys, group_labels, out_path):
    # fixed mid-game board for q value comparison
    fixed_board = np.array([
        [256, 128, 64, 32],
        [16, 8, 4, 2],
        [4, 2, 0, 0],
        [0, 0, 0, 0]
    ], dtype=np.int64)

    flat = fixed_board.flatten().astype(np.float32)
    state = np.zeros(16, dtype=np.float32)
    mask = flat > 0
    state[mask] = np.log2(flat[mask])

    action_names = ['Up', 'Down', 'Left', 'Right']

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for i, (key, label) in enumerate(zip(group_keys, group_labels)):
        ax = axes[i]

        agent = DQNAgent()
        agent.load(f'checkpoints/{key}.pth')

        state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            q_vals = agent.online_net(state_t).squeeze(0).cpu().numpy()

        best = int(np.argmax(q_vals))
        bar_colors = ['#cccccc'] * 4
        bar_colors[best] = '#ffd700'

        bars = ax.bar(action_names, q_vals, color=bar_colors, edgecolor='white', linewidth=1)
        q_range = max(q_vals) - min(q_vals) if max(q_vals) != min(q_vals) else 1
        for bar, val in zip(bars, q_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + q_range * 0.05,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_title(label, fontsize=9, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)

    # render the fixed board as a text annotation below the subplots
    board_lines = []
    for row in fixed_board:
        board_lines.append('  '.join(f'{v:>4}' if v > 0 else '   .' for v in row))
    board_text = 'fixed board state:\n' + '\n'.join(board_lines)
    fig.text(0.5, -0.01, board_text, ha='center', fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='#faf0e6', alpha=0.8))

    fig.suptitle('Q-Value Comparison on Fixed Board State (Part B)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {out_path}")


# generate animated gif replays for groups 5 through 8 and the random baseline
def generate_gameplay_gifs():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("pillow not installed, skipping gif generation. install with: pip install Pillow")
        return

    # tile color map following the classic 2048 theme
    tile_colors = {
        0: '#cdc1b4', 2: '#eee4da', 4: '#ede0c8', 8: '#f2b179',
        16: '#f59563', 32: '#f67c5f', 64: '#f65e3b', 128: '#edcf72',
        256: '#edcc61', 512: '#edc850', 1024: '#edc53f', 2048: '#edc22e',
    }
    default_tile_color = '#3c3a32'
    dark_text = '#776e65'
    light_text = '#f9f6f2'
    bg_color = '#bbada0'

    try:
        font_tile = ImageFont.load_default(size=24)
        font_header = ImageFont.load_default(size=14)
    except TypeError:
        font_tile = ImageFont.load_default()
        font_header = ImageFont.load_default()

    # render a single game frame as a pil image
    def render_frame(board, score, step_num):
        img = Image.new('RGB', (400, 440), '#faf8ef')
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), f'Score: {score}   Step: {step_num}',
                  fill=dark_text, font=font_header)

        grid_top = 40
        cell_size = 88
        gap = 6
        margin = int((400 - 4 * cell_size - 3 * gap) / 2)

        draw.rectangle(
            [margin - 4, grid_top, 400 - margin + 4, grid_top + 4 * cell_size + 3 * gap + 8],
            fill=bg_color)

        for r in range(4):
            for c in range(4):
                val = int(board[r, c])
                x = margin + c * (cell_size + gap)
                y = grid_top + 4 + r * (cell_size + gap)

                color = tile_colors.get(val, default_tile_color)
                draw.rectangle([x, y, x + cell_size, y + cell_size], fill=color)

                if val > 0:
                    text = str(val)
                    text_color = dark_text if val <= 4 else light_text
                    bbox = draw.textbbox((0, 0), text, font=font_tile)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                    tx = x + (cell_size - tw) // 2
                    ty = y + (cell_size - th) // 2
                    draw.text((tx, ty), text, fill=text_color, font=font_tile)

        return img

    # generate a gif for each double dqn group and the random baseline
    configs = [(gid, True, f'plots/part_b_replay_group_{gid}.gif') for gid in range(5, 9)]
    configs.append((None, False, 'plots/part_b_replay_random.gif'))

    for gid, use_ddqn, save_path in configs:
        env = Game2048()

        if gid is not None:
            agent = DQNAgent(use_double_dqn=use_ddqn)
            agent.load(f'checkpoints/group_{gid}.pth')
            agent.epsilon = 0.0

        state = env.reset()
        frames = [render_frame(env.board, env.score, 0)]
        done = False
        step = 0

        while not done and step < 500:
            if gid is not None:
                action = agent.select_action(state)
            else:
                action = np.random.randint(4)
            state, _, done, info = env.step(action)
            step += 1
            frames.append(render_frame(env.board, info['score'], step))

        # sample down to roughly 150 frames if the episode was long
        if len(frames) > 150:
            indices = np.linspace(0, len(frames) - 1, 150, dtype=int)
            frames = [frames[i] for i in indices]

        frames[0].save(save_path, save_all=True, append_images=frames[1:],
                       duration=200, loop=0)
        print(f"saved {save_path}")


# load checkpoints, evaluate groups 5 through 8 and the random baseline, generate all part b plots
def run_evaluation_b(num_eval_episodes=100):
    with open('results/part_b_training_results.pkl', 'rb') as f:
        training_results = pickle.load(f)

    os.makedirs('plots', exist_ok=True)

    keys = [f'group_{i}' for i in range(5, 9)] + ['random']
    labels = [
        'G5 DDQN, score only',
        'G6 DDQN, game over penalty',
        'G7 DDQN, invalid move penalty',
        'G8 DDQN, both penalties',
        'Random baseline',
    ]
    colors = ['#9467bd', '#8c564b', '#e377c2', '#17becf', '#7f7f7f']

    eval_results = {}

    for group_id in range(5, 9):
        key = f'group_{group_id}'
        agent = DQNAgent(use_double_dqn=True)
        agent.load(f'checkpoints/{key}.pth')
        scores, max_tiles = evaluate_agent(agent, num_eval_episodes)
        eval_results[key] = {'scores': scores, 'max_tiles': max_tiles}
        print(f"group {group_id}: avg score {np.mean(scores):.1f}, "
              f"avg max tile {np.mean(max_tiles):.1f}")

    scores, max_tiles = evaluate_random(num_eval_episodes)
    eval_results['random'] = {'scores': scores, 'max_tiles': max_tiles}
    print(f"random baseline: avg score {np.mean(scores):.1f}, "
          f"avg max tile {np.mean(max_tiles):.1f}")

    # trained groups only for action distribution and q value heatmap
    trained_keys = [f'group_{i}' for i in range(5, 9)]
    trained_labels = labels[:4]

    plot_avg_score(eval_results, keys, labels, colors, 'plots/part_b_avg_score.png')
    plot_avg_max_tile(eval_results, keys, labels, colors, 'plots/part_b_avg_max_tile.png')
    plot_reward_curves(training_results, keys, labels, colors, 'plots/part_b_reward_curves.png')
    plot_action_distribution(training_results, trained_keys, trained_labels,
                             'plots/part_b_action_distribution.png')
    plot_qvalue_heatmap(trained_keys, trained_labels, 'plots/part_b_qvalue_heatmap.png')
    generate_gameplay_gifs()

    with open('results/part_b_eval_results.pkl', 'wb') as f:
        pickle.dump(eval_results, f)

    print("\npart b evaluation complete, plots saved to plots/")
    return eval_results


if __name__ == '__main__':
    run_evaluation_b()
