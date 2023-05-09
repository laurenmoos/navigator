# naive implementation of ICM
import numpy as np
import torch
from torch import Tensor, nn
import pytorch_lightning as pl


class ForwardModel(nn.Module):
    def __init__(self, action_shape: int, state_latent: int):
        super().__init__()
        self.hidden = nn.Sequential(
            # subtract one since we do not consider the last state or the last action
            nn.Linear(action_shape + state_latent - 1, state_latent),
            nn.ReLU(inplace=True),
            nn.Linear(state_latent, state_latent),
            nn.ReLU(inplace=True),
            nn.Linear(state_latent, state_latent)
        )

    def forward(self, state_latent: Tensor, action: Tensor):
        x = torch.cat((action, state_latent), dim=-1)
        x = self.hidden(x)
        return x


class InverseModel(nn.Module):
    def __init__(self, state_latent: int, action_shape: int):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(state_latent * 2, state_latent),
            nn.ReLU(inplace=True),
            nn.Linear(state_latent, state_latent),
            nn.ReLU(inplace=True),
            nn.Linear(state_latent, action_shape)
        )

    def forward(self, state_latent: Tensor, next_state_latent: Tensor):
        cat = torch.cat((state_latent, next_state_latent), dim=-1)
        return self.input(cat)


class ICM(pl.LightningModule):
    """
    Note: this model will have to be altered to support recurrent data
    """
    def __init__(
            self,
            state_shape: int,
            action_shape: int,
            state_latent_size: int,
            policy_weight: float,
            reward_scale: float,
            weight: float
    ):
        super(ICM, self).__init__()

        self.policy_weight = policy_weight
        self.reward_scale = reward_scale
        self.weight = weight

        self.state_encoder = nn.Sequential(
            nn.Linear(state_shape, state_latent_size),
            nn.ReLU(inplace=True),
            nn.Linear(state_latent_size, state_latent_size),
            nn.ReLU(inplace=True),
            nn.Linear(state_latent_size, state_latent_size))

        self.forward_model = ForwardModel(action_shape, state_latent_size)
        self.inverse_model = InverseModel(state_latent_size, action_shape)

    @torch.no_grad()
    def __call__(self, state: Tensor, next_state: Tensor, action: Tensor):
        phi_state = self.state_encoder(state)
        phi_next_state = self.state_encoder(next_state)
        next_state_hat = self.forward_model(phi_state, action)
        action_hat = self.inverse_model(phi_state, phi_next_state)
        return phi_next_state, next_state_hat, action_hat

    @staticmethod
    def _transform_to_input(rewards: list, states: list, actions: list):
        vectorized_rewards = torch.FloatTensor(rewards)
        vectorized_states = torch.stack(states)
        vectorized_actions = torch.unsqueeze(torch.stack(actions), dim=1)

        return vectorized_states[:-1], vectorized_states[:-1], vectorized_actions, vectorized_rewards

    def oracle_reward(self, state, oracle_state, action) -> np.ndarray:
        next_state_latent, next_state_hat, _ = self.model(state, oracle_state, action)
        return self.reward_scale / 2 * (next_state_hat - next_state_latent).norm(2, dim=-1).pow(2)

    def temp_reward(self, rewards: list, states: list, actions: list) -> np.ndarray:
        states, next_states, actions, rewards = self._transform_to_input(rewards, states, actions)
        next_states_latent, next_states_hat, _ = self.model(states, next_states, actions[:-1])
        intrinsic_reward = self.reward_scale / 2 * (next_states_hat - next_states_latent).norm(2, dim=-1).pow(2)
        return intrinsic_reward

    def loss(self, states: Tensor, next_states: Tensor, actions: Tensor) -> Tensor:
        next_states_latent, next_states_hat, actions_hat = self.model(states, next_states, actions)
        forward_loss = 0.5 * (next_states_hat - next_states_latent.detach()).norm(2, dim=-1).pow(2).mean()
        dist = nn.PairwiseDistance(p=2)
        inverse_loss = dist(actions_hat, actions)
        return self.weight * forward_loss + (1 - self.weight) * inverse_loss

    def to(self, device: torch.device, dtype: torch.dtype):
        self.model.to(device, dtype)
