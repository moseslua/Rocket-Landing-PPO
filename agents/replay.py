"""Bounded NumPy replay; no persistent Python-object serialization."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ReplayBuffer:
    capacity: int
    observation_size: int
    action_size: int
    seed: int = 0

    def __post_init__(self):
        if self.capacity < 1 or self.observation_size < 1 or self.action_size < 1:
            raise ValueError("replay dimensions must be positive")
        self.observations = np.zeros((self.capacity, self.observation_size), np.float32)
        self.actions = np.zeros((self.capacity, self.action_size), np.float32)
        self.rewards = np.zeros(self.capacity, np.float32)
        self.next_observations = np.zeros_like(self.observations)
        self.dones = np.zeros(self.capacity, np.float32)
        self.costs = np.zeros((self.capacity, 3), np.float32)
        self.index = 0
        self.size = 0
        self.rng = np.random.default_rng(self.seed)

    def add(self, observation, action, reward, next_observation, done, costs=(0.0, 0.0, 0.0)):
        values = (observation, action, next_observation)
        if any(
            np.shape(value) != expected.shape
            for value, expected in zip(
                values,
                (self.observations[0], self.actions[0], self.next_observations[0]),
                strict=True,
            )
        ):
            raise ValueError("replay item has an invalid shape")
        if not all(np.isfinite(value).all() for value in (*values, reward, done)):
            raise ValueError("replay item contains nonfinite values")
        i = self.index
        self.observations[i], self.actions[i], self.rewards[i] = observation, action, reward
        self.next_observations[i], self.dones[i], self.costs[i] = next_observation, done, costs
        self.index = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if not 0 < batch_size <= self.size:
            raise ValueError("not enough replay samples")
        indices = self.rng.integers(0, self.size, size=batch_size)
        return {
            "observation": self.observations[indices],
            "action": self.actions[indices],
            "reward": self.rewards[indices],
            "next_observation": self.next_observations[indices],
            "done": self.dones[indices],
            "costs": self.costs[indices],
        }
