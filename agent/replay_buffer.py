import logging
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.utils.data import IterableDataset
from typing import T_co


class Batch(IterableDataset):

    def __getitem__(self, index) -> T_co:
        pass

    def __init__(self, generate_batch: Callable):
        self.generate_batch = generate_batch

    def __iter__(self):
        iterator = self.generate_batch()
        return iterator


@dataclass(frozen=False)
class Episode:
    rewards: list
    values: list
    states: list
    actions: list

    def __init__(self):
        self.rewards = []
        self.values = []
        self.states = []
        self.actions = []

    def update(self, reward, value, state, action):
        self.rewards.append(reward)
        self.values.append(value)
        self.states.append(state)

    def reset(self):
        # might want to do an optimized copy if it isn't slower
        self.rewards = []
        self.values = []
        self.states = []
        self.actions = []


@dataclass(frozen=False)
class MiniBatch:
    states: list
    next_states: list
    actions: list
    advs: list
    qvals: list
    logp: list

    def __init__(self):
        self.states = []
        self.next_states = []
        self.actions = []
        self.adv = []
        self.qvals = []
        self.logp = []

    def update_experience(self, state: torch.Tensor, next_state: torch.Tensor, action: float, logp: float):
        self.states.append(state)
        self.next_states.append(next_state)
        self.actions.append(action)
        self.logp.append(logp)

    def update_reward(self, adv: list, qval: list, ):
        self.adv.append(adv)
        self.qvals.append(qval)

    def data(self):
        return self.states, self.next_states, self.actions, self.adv, self.qvals, self.logp

    def reset(self):
        self.states = []
        self.next_states = []
        self.actions = []
        self.adv = []
        self.qvals = []
        self.logp = []
