import pytorch_lightning as pl
import torch
from torch import nn
from torch.distributions import Categorical
from torchsummary import summary


def create_mlp(input_shape, hidden_sizes: list):
    """
    Simple Multi-Layer Perceptron network
    """
    net_layers = [nn.Linear(input_shape, hidden_sizes[0]), nn.ReLU()]

    for i in range(len(hidden_sizes) - 1):
        net_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        net_layers.append(nn.ReLU())

    return nn.Sequential(*net_layers)


class Actor(pl.LightningModule):
    def __init__(self, shared, n_actions, n_hidden_units):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            shared,
            nn.Linear(128, n_actions)
        )

    def forward(self, state, **kwargs):
        logits = self.actor(state)
        pi = Categorical(logits=logits)
        actions = pi.sample()

        return pi, actions

    @staticmethod
    def get_log_prob(pi: Categorical, actions: torch.Tensor):
        return pi.log_prob(actions)


class ActorCritic(pl.LightningModule):
    def __init__(self, n_state, n_actions, n_hidden_units, actor_lr, critic_lr):
        super().__init__()
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        shared = create_mlp(n_state, [32, 32])

        self.actor = Actor(shared, n_actions, n_hidden_units)
        self.critic = nn.Sequential(
            shared,
            nn.Linear(n_hidden_units, 1)
        )

    @torch.no_grad()
    def __call__(self, state, **kwargs):
        pi, actions = self.actor(state)
        log_p = Actor.get_log_prob(pi, actions)
        value = self.critic(state)

        return pi, actions, log_p, value

    def get_log_prob(self, pi, actions: torch.Tensor) -> torch.Tensor:
        logp =  self.actor.get_log_prob(pi, actions)
        return logp
