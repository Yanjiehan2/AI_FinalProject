# DQN Agent for 2048

A PyTorch implementation of a Deep Q-Network (DQN) agent trained to play the 2048 puzzle game. Four experimental reward-shaping configurations are compared against a random baseline to study the effect of game-over and invalid-move penalties on learning performance.

## Project Structure

```
.
├── game_env.py     # 2048 game environment
├── dqn_agent.py    # DQN agent with replay buffer and target network
├── train.py        # training loop for all 4 groups and random baseline
├── evaluate.py     # evaluation runs and matplotlib plots
├── main.py         # entry point: runs training then evaluation
├── checkpoints/    # saved model weights per group (created at runtime)
├── results/        # serialised training and evaluation data (created at runtime)
└── plots/          # PNG figures (created at runtime)
```

## Installation

Python 3.8 or later is required.

```bash
pip install torch numpy matplotlib
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

Each group trains a separate DQN agent for 10000 episodes. The base reward at every step is the score increment produced by the move (i.e. the sum of merged tile values). Penalties are added on top.

| Group | Game-over penalty | Invalid-move penalty | Description |
|-------|:-----------------:|:--------------------:|-------------|
| 1     | none              | none                 | Score increment only; no shaping |
| 2     | −100              | none                 | Discourages ending the game early |
| 3     | none              | −5                   | Discourages wasting moves that do not change the board |
| 4     | −100              | −5                   | Both penalties combined |

### Random baseline

A random agent that selects one of the four actions uniformly at random without any learning. It is run for 10000 episodes and included in all evaluation plots as a lower-bound reference.

## Outputs

After a full run the following files are produced:

- `checkpoints/group_<N>.pth` — final model checkpoint for each trained group
- `checkpoints/group_<N>_ep<E>.pth` — periodic checkpoint saved every 500 episodes
- `results/group_<N>_partial.pkl` — partial training data saved alongside each periodic checkpoint
- `results/training_results.pkl` — complete per-episode rewards, scores, and max tiles for all groups
- `results/eval_results.pkl` — evaluation scores and max tiles for all groups
- `plots/avg_score.png` — bar chart of mean score over 100 evaluation episodes
- `plots/avg_max_tile_log2.png` — bar chart of mean log2 of the highest tile reached
- `plots/reward_curves.png` — smoothed training reward curves for all four groups

## Resuming Training

If training is interrupted, re-running `python train.py` (or `python main.py`) will automatically detect the latest periodic checkpoint for each group and resume from that episode. The saved epsilon value and partial results are restored, so no progress is lost.

## Architecture

- **State**: 16-dimensional vector (flattened 4×4 board), each tile encoded as log2 of its value (0 tiles stay 0)
- **Network**: fully connected, 16 → 256 → 256 → 256 → 4, ReLU activations (three hidden layers)
- **Replay buffer**: 100 000 transitions, minibatch size 64
- **Warmup**: gradient updates are withheld until the replay buffer contains at least 1 000 transitions; the agent still acts with ε-greedy during warmup
- **Optimiser**: Adam, learning rate 1e-4
- **Loss**: MSE between predicted Q-values and Bellman targets
- **Exploration**: ε-greedy, ε decays from 1.0 to 0.01 (decay factor 0.9999 per update step)
- **Evaluation epsilon**: fixed at 0.05 during evaluation to prevent infinite loops on undertrained models; each evaluation episode is also capped at 2000 steps
- **Target network**: hard sync every 100 update steps
- **Checkpoints**: saved every 500 episodes per group, including model weights, episode number, and current ε
