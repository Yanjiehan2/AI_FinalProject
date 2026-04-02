import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


# fully connected q network with three hidden layers of 256 units each
class QNetwork(nn.Module):

    # build layers mapping state size to action size through three relu hidden layers
    def __init__(self, state_size=16, hidden_size=256, action_size=4):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x):
        return self.net(x)


# circular buffer storing transitions for experience replay
class ReplayBuffer:

    # create a deque with fixed maximum capacity
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    # append a single transition tuple to the buffer
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # sample a random minibatch and return stacked numpy arrays
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


# dqn and double dqn agent with replay buffer warmup, epsilon greedy, and periodic target sync
class DQNAgent:

    # initialize both networks, optimizer, replay buffer, and all hyperparameters including double dqn flag
    def __init__(
        self,
        state_size=16,
        action_size=4,
        lr=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.9999,
        buffer_size=100000,
        batch_size=64,
        target_update_freq=100,
        warmup_size=1000,
        use_double_dqn=False
    ):
        self.action_size = action_size
        self.use_double_dqn = use_double_dqn
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.warmup_size = warmup_size
        self.steps = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.online_net = QNetwork(state_size, 256, action_size).to(self.device)
        self.target_net = QNetwork(state_size, 256, action_size).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(buffer_size)

    # choose a random action with probability epsilon, otherwise pick the greedy action
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    # sample a minibatch, compute td targets, and update online network weights
    def update(self):
        if len(self.replay_buffer) < self.warmup_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        current_q = self.online_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # use online net to select actions and target net to evaluate when double dqn is on
            if self.use_double_dqn:
                best_actions = self.online_net(next_states_t).argmax(1, keepdim=True)
                next_q = self.target_net(next_states_t).gather(1, best_actions).squeeze(1)
            else:
                next_q = self.target_net(next_states_t).max(1)[0]
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1

        # sync target network weights every target_update_freq steps
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()

    # serialize network weights, optimizer state, epsilon, episode, and algorithm flag to a file
    def save(self, path, episode=None):
        torch.save({
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode': episode,
            'use_double_dqn': self.use_double_dqn
        }, path)

    # restore network weights, optimizer state, epsilon, and algorithm flag, then return episode number
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.online_net.load_state_dict(checkpoint['online_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.use_double_dqn = checkpoint.get('use_double_dqn', False)
        return checkpoint.get('episode', 0)
