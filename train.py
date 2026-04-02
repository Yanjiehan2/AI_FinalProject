import os
import pickle
import numpy as np
from game_env import Game2048
from dqn_agent import DQNAgent


# scan the checkpoints directory and return the path and episode of the latest periodic save
def find_latest_checkpoint(group_id):
    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        return None, 0
    prefix = f'group_{group_id}_ep'
    matches = []
    for fname in os.listdir(checkpoint_dir):
        if fname.startswith(prefix) and fname.endswith('.pth'):
            try:
                ep = int(fname[len(prefix):-4])
                matches.append((ep, os.path.join(checkpoint_dir, fname)))
            except ValueError:
                pass
    if not matches:
        return None, 0
    matches.sort()
    return matches[-1][1], matches[-1][0]


# save agent weights and partial training results together at a milestone episode
def save_checkpoint(agent, group_id, episode, rewards, scores, max_tiles, action_counts=None):
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    agent.save(f'checkpoints/group_{group_id}_ep{episode}.pth', episode=episode)
    partial = {
        'rewards': rewards, 'scores': scores, 'max_tiles': max_tiles,
        'action_counts': action_counts
    }
    with open(f'results/group_{group_id}_partial.pkl', 'wb') as f:
        pickle.dump(partial, f)


# collect reward and score data for a purely random agent over num_episodes episodes
def run_random_baseline(num_episodes):
    env = Game2048()
    rewards = []
    scores = []
    max_tiles = []

    for ep in range(num_episodes):
        env.reset()
        total_reward = 0
        done = False

        while not done:
            action = np.random.randint(4)
            _, reward, done, info = env.step(action)
            total_reward += reward

        rewards.append(total_reward)
        scores.append(info['score'])
        max_tiles.append(int(env.board.max()))

        if (ep + 1) % 1000 == 0:
            print(f"  random baseline episode {ep + 1}/{num_episodes}, "
                  f"avg score {np.mean(scores[-1000:]):.1f}")

    return {'rewards': rewards, 'scores': scores, 'max_tiles': max_tiles}


# train a single dqn agent for num_episodes with the given penalty and algorithm configuration
def train_group(group_id, game_over_penalty=0, invalid_move_penalty=0,
                num_episodes=5000, use_double_dqn=False):
    env = Game2048()
    agent = DQNAgent(use_double_dqn=use_double_dqn)
    rewards = []
    scores = []
    max_tiles = []
    action_counts = []
    max_steps_per_episode = 5000
    checkpoint_interval = 500

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # resume from the latest periodic checkpoint if one exists for this group
    start_episode = 0
    ckpt_path, ckpt_episode = find_latest_checkpoint(group_id)
    if ckpt_path:
        agent.load(ckpt_path)
        start_episode = ckpt_episode
        partial_path = f'results/group_{group_id}_partial.pkl'
        if os.path.exists(partial_path):
            with open(partial_path, 'rb') as f:
                partial = pickle.load(f)
            rewards = partial['rewards']
            scores = partial['scores']
            max_tiles = partial['max_tiles']
            action_counts = partial.get('action_counts', [])
        print(f"  resumed from episode {start_episode}, epsilon {agent.epsilon:.4f}")

    for ep in range(start_episode, num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        episode_actions = np.zeros(4, dtype=np.int32)

        while not done and steps < max_steps_per_episode:
            action = agent.select_action(state)
            episode_actions[action] += 1
            next_state, score_increment, done, info = env.step(action)

            # compute shaped reward by adding configured penalties to the score increment
            reward = float(score_increment)
            if done:
                reward += game_over_penalty
            if info['invalid_move']:
                reward += invalid_move_penalty

            agent.store_transition(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        scores.append(info['score'])
        max_tiles.append(int(env.board.max()))
        action_counts.append(episode_actions.tolist())

        # save a periodic checkpoint and partial results every checkpoint_interval episodes
        if (ep + 1) % checkpoint_interval == 0:
            save_checkpoint(agent, group_id, ep + 1, rewards, scores, max_tiles, action_counts)
            print(f"  group {group_id} episode {ep + 1}/{num_episodes}, "
                  f"avg score {np.mean(scores[-checkpoint_interval:]):.1f}, "
                  f"epsilon {agent.epsilon:.4f}, checkpoint saved")

    agent.save(f'checkpoints/group_{group_id}.pth', episode=num_episodes)
    return {
        'rewards': rewards, 'scores': scores, 'max_tiles': max_tiles,
        'action_counts': action_counts
    }


# train additional groups to study how penalty magnitude affects learning
def train_parameter_study(num_episodes):
    # only train groups that are not duplicates of existing main groups
    ps_configs = {
        'ps1': (-50, 0),
        'ps3': (-200, 0),
        'ps4': (0, -2),
        'ps6': (0, -10),
    }

    results = {}
    for ps_id, (go_penalty, inv_penalty) in ps_configs.items():
        print(f"\ntraining parameter study group {ps_id}, "
              f"game over penalty {go_penalty}, "
              f"invalid move penalty {inv_penalty}")
        results[ps_id] = train_group(ps_id, go_penalty, inv_penalty, num_episodes)

    # reuse group 2 results as ps2 and group 3 results as ps5 to avoid retraining
    with open('results/training_results.pkl', 'rb') as f:
        main_results = pickle.load(f)
    results['ps2'] = main_results['group_2']
    results['ps5'] = main_results['group_3']

    os.makedirs('results', exist_ok=True)
    with open('results/param_study_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("\nparameter study complete, results saved to results/param_study_results.pkl")
    return results


# run training for all eight groups, the random baseline, and the parameter study
def train_all_groups(num_episodes=5000):
    # each entry maps group id to game over penalty, invalid move penalty, and double dqn flag
    group_configs = {
        1: (0, 0, False),
        2: (-100, 0, False),
        3: (0, -5, False),
        4: (-100, -5, False),
        5: (0, 0, True),
        6: (-100, 0, True),
        7: (0, -5, True),
        8: (-100, -5, True),
    }

    results = {}

    for group_id, (go_penalty, inv_penalty, use_ddqn) in group_configs.items():
        algo = 'double dqn' if use_ddqn else 'dqn'
        print(f"\ntraining group {group_id} ({algo}), "
              f"game over penalty {go_penalty}, "
              f"invalid move penalty {inv_penalty}")
        results[f'group_{group_id}'] = train_group(
            group_id, go_penalty, inv_penalty, num_episodes, use_double_dqn=use_ddqn
        )

    print("\nrunning random baseline")
    results['random'] = run_random_baseline(num_episodes)

    os.makedirs('results', exist_ok=True)
    with open('results/training_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("\ntraining complete, results saved to results/training_results.pkl")

    # run the parameter study after main training is complete
    train_parameter_study(num_episodes)

    return results


if __name__ == '__main__':
    train_all_groups()
