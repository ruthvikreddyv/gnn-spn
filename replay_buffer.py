"""
replay_buffer.py  –  Dual-purpose Replay Buffer
=================================================
Supports both:
  1. Supervised phase  – stores (X_t, A_t, Y*_t) tuples
  2. PPO rollout phase – stores (X_t, A_t, actions, log_probs, rewards, values, dones)

GAE-λ advantage computation included for PPO phase.
"""

import numpy as np


class SupervisedBuffer:
    """Replay buffer for supervised pre-training."""

    def __init__(self, capacity=10_000, n_links=60,
                 n_features=82, n_actions=60, n_veh=20):
        self.capacity   = capacity
        self.n_links    = n_links
        self.n_features = n_features
        self.n_actions  = n_actions
        self.n_veh      = n_veh

        self.X      = np.zeros((capacity, n_links, n_features), dtype=np.float32)
        self.A      = np.zeros((capacity, n_veh, n_veh),         dtype=np.float32)
        self.labels = np.zeros((capacity, n_links, n_actions),   dtype=np.float32)

        self.count   = 0
        self.current = 0

    def add(self, X, A, labels):
        self.X[self.current]      = X[:self.n_links]
        self.A[self.current]      = A[:self.n_veh, :self.n_veh]
        self.labels[self.current] = labels[:self.n_links]
        self.count   = min(self.count + 1, self.capacity)
        self.current = (self.current + 1) % self.capacity

    def sample(self, batch_size=64):
        idx = np.random.randint(0, self.count, size=batch_size)
        return self.X[idx], self.A[idx], self.labels[idx]

    def ready(self, min_size=500):
        return self.count >= min_size

    def __len__(self):
        return self.count


class RolloutBuffer:
    """On-policy rollout buffer for PPO fine-tuning with GAE-λ."""

    def __init__(self, capacity=60, n_links=60,
                 n_features=82, n_veh=20,
                 gamma=0.99, gae_lambda=0.95):
        self.capacity   = capacity
        self.n_links    = n_links
        self.n_features = n_features
        self.n_veh      = n_veh
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self._reset()

    def _reset(self):
        self.X         = np.zeros((self.capacity, self.n_links,
                                   self.n_features), dtype=np.float32)
        self.A         = np.zeros((self.capacity, self.n_veh,
                                   self.n_veh),    dtype=np.float32)
        self.actions   = np.zeros((self.capacity, self.n_links), dtype=np.int32)
        self.log_probs = np.zeros((self.capacity, self.n_links), dtype=np.float32)
        self.rewards   = np.zeros(self.capacity,                 dtype=np.float32)
        self.values    = np.zeros(self.capacity,                 dtype=np.float32)
        self.dones     = np.zeros(self.capacity,                 dtype=np.float32)
        self.ptr = 0

    def add(self, X, A, actions, log_probs, reward, value, done):
        t = self.ptr
        self.X[t]         = X
        self.A[t]         = A
        self.actions[t]   = actions
        self.log_probs[t] = log_probs
        self.rewards[t]   = reward
        self.values[t]    = value
        self.dones[t]     = float(done)
        self.ptr += 1

    def ready(self):
        return self.ptr >= self.capacity

    def compute_gae(self, last_value=0.0):
        T = self.ptr
        adv     = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)
        gae     = 0.0
        next_v  = last_value
        for t in reversed(range(T)):
            delta = (self.rewards[t]
                     + self.gamma * next_v * (1-self.dones[t])
                     - self.values[t])
            gae   = delta + self.gamma * self.gae_lambda * (1-self.dones[t]) * gae
            adv[t] = gae
            next_v = self.values[t]
        returns = adv + self.values[:T]
        # normalise advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return (self.X[:T], self.A[:T], self.actions[:T],
                self.log_probs[:T], returns, adv)

    def clear(self):
        self._reset()
