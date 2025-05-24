import random
from collections import deque
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

from utils import calculate_score


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int = 2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.model(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(list, zip(*batch))
        return (torch.stack(state),
                torch.tensor(action, dtype=torch.long),
                torch.tensor(reward, dtype=torch.float),
                torch.stack(next_state),
                torch.tensor(done, dtype=torch.float))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_dim: int, hidden_dim: int = 64, lr: float = 5e-4,
                 gamma: float = 0.99, epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05, epsilon_decay: float = 0.995,
                 buffer_size: int = 10000, batch_size: int = 32,
                 target_update: int = 10, double_dqn: bool = True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = QNetwork(state_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.memory = ReplayBuffer(buffer_size)
        self.steps = 0
        self.double_dqn = double_dqn

    def select_action(self, state: torch.Tensor) -> int:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        if random.random() < self.epsilon:
            return random.randrange(2)
        with torch.no_grad():
            q_values = self.policy_net(state.to(self.device))
            return int(torch.argmax(q_values).item())

    def store(self, state, action, reward, next_state, done):
        self.memory.push(state.cpu(), action, reward, next_state.cpu(), done)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        next_states = next_states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.policy_net(next_states).argmax(1)
                next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            else:
                next_q = self.target_net(next_states).max(1)[0]
        target = rewards + (1 - dones) * self.gamma * next_q
        loss = nn.functional.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


class RLRanker:
    """使用DQN根据用户偏好对房产进行排序"""

    def __init__(self, criteria: dict, preferences: dict, state_dim: int):
        self.criteria = criteria
        self.weights = preferences.get('weights', {})
        eps_start = preferences.get('epsilon_start', 1.0)
        eps_end = preferences.get('epsilon_end', 0.05)
        eps_decay = preferences.get('epsilon_decay', 0.995)
        self.diversity_weight = preferences.get('diversity_weight', 0.1)
        self.agent = DQNAgent(state_dim, lr=5e-4,
                              epsilon_start=eps_start,
                              epsilon_end=eps_end,
                              epsilon_decay=eps_decay,
                              double_dqn=True)

    def train(self, houses: List[dict], episodes: int = 100, verbose: bool = False) -> list:
        """训练 DQN 并返回每个 episode 的平均加权得分"""
        episode_scores = []
        for ep in range(episodes):
            total = 0.0
            visited = set()
            random.shuffle(houses)
            for i, house in enumerate(houses):
                state = torch.tensor(house['embedding'], dtype=torch.float)
                action = self.agent.select_action(state)
                base = calculate_score(house, self.criteria, self.weights) if action == 1 else 0.0
                bonus = 0.0
                if action == 1 and house['id'] not in visited:
                    bonus = self.diversity_weight * (1.0 - len(visited) / len(houses))
                    visited.add(house['id'])
                reward = base + bonus
                done = i == len(houses) - 1
                next_state = torch.tensor(houses[(i + 1) % len(houses)]['embedding'], dtype=torch.float)
                self.agent.store(state, action, reward, next_state, float(done))
                self.agent.train_step()
                total += base
            avg_score = total / max(1, len(houses))
            episode_scores.append(avg_score)
            if verbose:
                print(f'Episode {ep + 1}: score={avg_score:.2f}')
        return episode_scores

    def rank(self, houses: List[dict]) -> List[dict]:
        scored = []
        for house in houses:
            state = torch.tensor(house['embedding'], dtype=torch.float)
            with torch.no_grad():
                q = self.agent.policy_net(state)
            scored.append((house, q[1].item()))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [h for h, _ in scored]

    def rank_with_scores(self, houses: List[dict]) -> List[tuple]:
        """返回按得分排序的 (house, score) 列表"""
        scored = []
        for house in houses:
            state = torch.tensor(house['embedding'], dtype=torch.float)
            with torch.no_grad():
                q = self.agent.policy_net(state)
            scored.append((house, q[1].item()))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def update_weights(self, weights: dict) -> None:
        """在训练过程中动态调整权重"""
        self.weights = weights

