# DQN Agent for 2048

A PyTorch implementation of DQN and Double DQN agents trained to play the 2048 puzzle game. The project is split into three independent parts. Eight experimental configurations (four reward shaping setups times two algorithms) are compared against a random baseline. An additional parameter study examines how penalty magnitude affects learning.

## Project Structure

```
.
├── game_env.py        # 2048 game environment
├── dqn_agent.py       # DQN and Double DQN agent with replay buffer and target network
├── train_part_a.py    # Part A training: DQN Groups 1–4
├── eval_part_a.py     # Part A evaluation and plots for Groups 1–4
├── train_part_b.py    # Part B training: Double DQN Groups 5–8 and random baseline
├── eval_part_b.py     # Part B evaluation and plots for Groups 5–8 and random baseline
├── train_part_c.py    # Part C training: parameter study baseline and PS1–PS6
├── eval_part_c.py     # Part C evaluation and parameter study plots
├── checkpoints/       # saved model weights per group (created at runtime)
├── results/           # serialised training and evaluation data (created at runtime)
└── plots/             # PNG figures and GIF replays (created at runtime)
```

## Installation

Python 3.8 or later is required.

```bash
pip install torch numpy matplotlib Pillow
```

## Running

### Part A: DQN Groups 1–4

```bash
# training (default 10000 episodes; pass a number to override, e.g. 500 for a quick test)
python train_part_a.py
python train_part_a.py 500

# evaluation and plots (requires completed Part A training)
python eval_part_a.py
```

### Part B: Double DQN Groups 5–8 and random baseline

```bash
python train_part_b.py
python train_part_b.py 500

python eval_part_b.py
```

### Part C: Parameter study

```bash
python train_part_c.py
python train_part_c.py 500

python eval_part_c.py
```

## Experiment Groups

### Part A — Standard DQN (Groups 1–4)

Each group trains a DQN agent for 10 000 episodes. The base reward is the score increment from each move (sum of merged tile values). Penalties are added on top.

| Group | Algorithm | Game-over penalty | Invalid-move penalty | Description |
|-------|:---------:|:-----------------:|:--------------------:|-------------|
| 1     | DQN       | none              | none                 | Score increment only, no shaping |
| 2     | DQN       | −100              | none                 | Discourages ending the game early |
| 3     | DQN       | none              | −5                   | Discourages wasting moves that do not change the board |
| 4     | DQN       | −100              | −5                   | Both penalties combined |

### Part B — Double DQN (Groups 5–8) and random baseline

| Group  | Algorithm  | Game-over penalty | Invalid-move penalty | Description |
|--------|:----------:|:-----------------:|:--------------------:|-------------|
| 5      | Double DQN | none              | none                 | Same as Group 1 with Double DQN |
| 6      | Double DQN | −100              | none                 | Same as Group 2 with Double DQN |
| 7      | Double DQN | none              | −5                   | Same as Group 3 with Double DQN |
| 8      | Double DQN | −100              | −5                   | Same as Group 4 with Double DQN |
| Random | —          | —                 | —                    | Uniform random action selection, lower-bound reference |

### Part C — Parameter study (PS baseline, PS1–PS6)

All groups use standard DQN. One penalty is varied at a time while the other is held at zero.

| Group       | Game-over penalty | Invalid-move penalty | Notes |
|-------------|:-----------------:|:--------------------:|-------|
| PS baseline | none              | none                 | No-penalty reference, trained fresh |
| PS1         | −50               | none                 | |
| PS2         | −100              | none                 | |
| PS3         | −200              | none                 | |
| PS4         | none              | −2                   | |
| PS5         | none              | −5                   | |
| PS6         | none              | −10                  | |

## Outputs

### Part A outputs

- `results/part_a_training_results.pkl`
- `results/part_a_eval_results.pkl`
- `plots/part_a_avg_score.png`
- `plots/part_a_avg_max_tile.png`
- `plots/part_a_reward_curves.png`
- `plots/part_a_action_distribution.png` — 1×4 grid
- `plots/part_a_qvalue_heatmap.png` — 1×4 grid
- `plots/part_a_replay_group_<1–4>.gif`

### Part B outputs

- `results/part_b_training_results.pkl`
- `results/part_b_eval_results.pkl`
- `plots/part_b_avg_score.png`
- `plots/part_b_avg_max_tile.png`
- `plots/part_b_reward_curves.png`
- `plots/part_b_action_distribution.png` — 1×4 grid (trained groups only)
- `plots/part_b_qvalue_heatmap.png` — 1×4 grid (trained groups only)
- `plots/part_b_replay_group_<5–8>.gif`
- `plots/part_b_replay_random.gif`

### Part C outputs

- `results/part_c_training_results.pkl`
- `results/part_c_eval_results.pkl`
- `plots/part_c_parameter_study.png` — line plots of score vs penalty magnitude
- `plots/part_c_avg_score.png`
- `plots/part_c_reward_curves.png`

## Resuming Training

If training is interrupted, re-running any training script will automatically detect the latest periodic checkpoint for each group and resume from that episode. The saved epsilon value, action counts, and partial results are restored, so no progress is lost.

## Architecture

- **State**: 16-dimensional vector (flattened 4×4 board), each tile encoded as log2 of its value (0 tiles stay 0)
- **Network**: fully connected, 16 → 256 → 256 → 256 → 4, ReLU activations (three hidden layers)
- **Algorithms**: standard DQN uses the target network to both select and evaluate the next action; Double DQN uses the online network to select the best next action and the target network to evaluate it, reducing overestimation bias
- **Replay buffer**: 100 000 transitions, minibatch size 64
- **Warmup**: gradient updates are withheld until the replay buffer contains at least 1 000 transitions; the agent still acts with epsilon-greedy during warmup
- **Optimiser**: Adam, learning rate 1e-4
- **Loss**: MSE between predicted Q-values and Bellman targets
- **Exploration**: epsilon-greedy, epsilon decays from 1.0 to 0.01 (decay factor 0.9999 per update step)
- **Evaluation epsilon**: fixed at 0.05 during evaluation to prevent infinite loops on undertrained models; each evaluation episode is also capped at 2000 steps
- **Target network**: hard sync every 100 update steps
- **Checkpoints**: saved every 500 episodes per group, including model weights, episode number, and current epsilon
