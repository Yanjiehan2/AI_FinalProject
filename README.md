# DQN Agent for 2048

A PyTorch implementation of DQN and Double DQN agents trained to play the 2048 puzzle game. Eight experimental configurations (four reward shaping setups times two algorithms) are compared against a random baseline. An additional parameter study examines how penalty magnitude affects learning.

## Project Structure

```
.
├── game_env.py     # 2048 game environment
├── dqn_agent.py    # DQN and Double DQN agent with replay buffer and target network
├── train.py        # training loop for all 8 groups, random baseline, and parameter study
├── evaluate.py     # evaluation, plotting, GIF generation, and parameter study analysis
├── main.py         # entry point: runs training then evaluation
├── checkpoints/    # saved model weights per group (created at runtime)
├── results/        # serialised training and evaluation data (created at runtime)
└── plots/          # PNG figures and GIF replays (created at runtime)
```

## Installation

Python 3.8 or later is required.

```bash
pip install torch numpy matplotlib Pillow
```

## Running

### Full pipeline (training + evaluation)

```bash
python main.py
```

### Training only

```bash
python train.py
```

### Evaluation and plotting only (requires completed training)

```bash
python evaluate.py
```

## Experiment Groups

Each group trains a separate agent for 10000 episodes. The base reward at every step is the score increment produced by the move (i.e. the sum of merged tile values). Penalties are added on top. Groups 1 through 4 use standard DQN; Groups 5 through 8 use Double DQN.

| Group | Algorithm  | Game-over penalty | Invalid-move penalty | Description |
|-------|:----------:|:-----------------:|:--------------------:|-------------|
| 1     | DQN        | none              | none                 | Score increment only, no shaping |
| 2     | DQN        | −100              | none                 | Discourages ending the game early |
| 3     | DQN        | none              | −5                   | Discourages wasting moves that do not change the board |
| 4     | DQN        | −100              | −5                   | Both penalties combined |
| 5     | Double DQN | none              | none                 | Same as Group 1 with Double DQN |
| 6     | Double DQN | −100              | none                 | Same as Group 2 with Double DQN |
| 7     | Double DQN | none              | −5                   | Same as Group 3 with Double DQN |
| 8     | Double DQN | −100              | −5                   | Same as Group 4 with Double DQN |

### Random baseline

A random agent that selects one of the four actions uniformly at random without any learning. It is run for 10000 episodes and included in all evaluation plots as a lower-bound reference.

### Parameter study (PS1 through PS6)

Additional groups that vary a single penalty magnitude while keeping the other at zero, using standard DQN to isolate the effect of penalty size.

| Group | Game-over penalty | Invalid-move penalty | Notes |
|-------|:-----------------:|:--------------------:|-------|
| PS1   | −50               | none                 | |
| PS2   | −100              | none                 | Same as Group 2, results reused |
| PS3   | −200              | none                 | |
| PS4   | none              | −2                   | |
| PS5   | none              | −5                   | Same as Group 3, results reused |
| PS6   | none              | −10                  | |

## Outputs

After a full run the following files are produced:

- `checkpoints/group_<N>.pth` — final model checkpoint for each trained group (1 through 8 and PS groups)
- `checkpoints/group_<N>_ep<E>.pth` — periodic checkpoint saved every 500 episodes
- `results/group_<N>_partial.pkl` — partial training data saved alongside each periodic checkpoint
- `results/training_results.pkl` — complete per-episode rewards, scores, max tiles, and action counts for all groups
- `results/param_study_results.pkl` — training results for the parameter study groups
- `results/eval_results.pkl` — evaluation scores and max tiles for all groups
- `plots/avg_score.png` — bar chart of mean score over 100 evaluation episodes
- `plots/avg_max_tile_log2.png` — bar chart of mean log2 of the highest tile reached
- `plots/reward_curves.png` — smoothed training reward curves for all eight trained groups
- `plots/action_distribution.png` — stacked area charts showing per-action proportions during training
- `plots/qvalue_heatmap.png` — Q-value bar charts for a fixed board state across all groups
- `plots/replay_group_<N>.gif` — animated gameplay replay for each trained group
- `plots/replay_random.gif` — animated gameplay replay for the random baseline
- `plots/parameter_study.png` — line plots of score vs penalty magnitude

## Resuming Training

If training is interrupted, re-running `python train.py` (or `python main.py`) will automatically detect the latest periodic checkpoint for each group and resume from that episode. The saved epsilon value, action counts, and partial results are restored, so no progress is lost.

## Architecture

- **State**: 16-dimensional vector (flattened 4x4 board), each tile encoded as log2 of its value (0 tiles stay 0)
- **Network**: fully connected, 16 → 256 → 256 → 256 → 4, ReLU activations (three hidden layers)
- **Algorithms**: standard DQN (Groups 1 through 4) uses the target network to both select and evaluate the next action; Double DQN (Groups 5 through 8) uses the online network to select the best next action and the target network to evaluate it, reducing overestimation bias
- **Replay buffer**: 100 000 transitions, minibatch size 64
- **Warmup**: gradient updates are withheld until the replay buffer contains at least 1 000 transitions; the agent still acts with epsilon-greedy during warmup
- **Optimiser**: Adam, learning rate 1e-4
- **Loss**: MSE between predicted Q-values and Bellman targets
- **Exploration**: epsilon-greedy, epsilon decays from 1.0 to 0.01 (decay factor 0.9999 per update step)
- **Evaluation epsilon**: fixed at 0.05 during evaluation to prevent infinite loops on undertrained models; each evaluation episode is also capped at 2000 steps
- **Target network**: hard sync every 100 update steps
- **Checkpoints**: saved every 500 episodes per group, including model weights, episode number, and current epsilon
