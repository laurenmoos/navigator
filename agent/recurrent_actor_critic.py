# https://gitlab.com/ngoodger/ppo_lstm/-/blob/master/recurrent_ppo.ipynb
# this link contains an implementation of a recurrent policy but with a continuous action space 
import pytorch_lightning as pl
import torch
from torch import nn
from torch.distributions import Categorical

'''
self.layer_lstm = nn.LSTM(state_dim, hp.hidden_size, num_layers=hp.recurrent_layers)
self.layer_hidden = nn.Linear(hp.hidden_size, hp.hidden_size)

self.hidden_cell = None
        '''

class LambdaModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[0]


# LSTM() returns tuple of (tensor, (recurrent state))
# class Extract(nn.Module):
#     def __init__(self):
#         super(Extract, self).__init__()
#
#     def forward(self, x):
#         # Output shape (batch, features, hidden)
#         tensor, _ = x
#         # Reshape shape (batch, hidden)
#         return tensor[:, -1, :]


# TODO: should this share weights?
def create_lstm(state_dim, hidden_size=32, recurrent_layers=8):
    return nn.LSTM(state_dim, hidden_size, num_layers=recurrent_layers)


class Actor(pl.LightningModule):

    def __init__(self, shared, n_actions, n_hidden_units):
        super(Actor, self).__init__()
        self.shared = shared
        self.n_actions = n_actions

        self.actor_net = nn.Sequential(
            self.shared,
            LambdaModule(),
            nn.Linear(32, 32),
            nn.Linear(32, n_actions)
        )

        # self.action_dim = action_dim
        # self.continuous_action_space = continuous_action_space
        # self.log_std_dev = nn.Parameter(init_log_std_dev * torch.ones((action_dim), dtype=torch.float),
        #                                 requires_grad=trainable_std_dev)
        # self.covariance_eye = torch.eye(self.action_dim).unsqueeze(0)
        # self.hidden_cell = None

    def forward(self, state, **kwargs):

        logits = self.actor_net(state).squeeze().squeeze()

        pi = Categorical(logits=logits)
        actions = pi.sample()

        return pi, actions

    @staticmethod
    def get_log_prob(pi: Categorical, actions: torch.Tensor):
        return pi.log_prob(actions)

    # def forward(self, state, terminal=None):
    #     batch_size = state.shape[1]
    #     device = state.device
    #     if self.hidden_cell is None or batch_size != self.hidden_cell[0].shape[1]:
    #         self.get_init_state(batch_size, device)
    #     if terminal is not None:
    #         self.hidden_cell = [value * (1. - terminal).reshape(1, batch_size, 1) for value in self.hidden_cell]
    #     _, self.hidden_cell = self.lstm(state, self.hidden_cell)
    #     hidden_out = F.elu(self.layer_hidden(self.hidden_cell[0][-1]))
    #     policy_logits_out = self.layer_policy_logits(hidden_out)
    #     if self.continuous_action_space:
    #         cov_matrix = self.covariance_eye.to(device).expand(batch_size, self.action_dim,
    #                                                            self.action_dim) * torch.exp(self.log_std_dev.to(device))
    #         # We define the distribution on the CPU since otherwise operations fail with CUDA illegal memory access error.
    #         policy_dist = torch.distributions.multivariate_normal.MultivariateNormal(policy_logits_out.to("cpu"),
    #                                                                                  cov_matrix.to("cpu"))
    #     else:
    #         policy_dist = distributions.Categorical(F.softmax(policy_logits_out, dim=1).to("cpu"))
    #     return policy_dist


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()

    # def forward(self, state, terminal=None):
    #     batch_size = state.shape[1]
    #     device = state.device
    #     if self.hidden_cell is None or batch_size != self.hidden_cell[0].shape[1]:
    #         self.get_init_state(batch_size, device)
    #     if terminal is not None:
    #         self.hidden_cell = [value * (1. - terminal).reshape(1, batch_size, 1) for value in self.hidden_cell]
    #     _, self.hidden_cell = self.layer_lstm(state, self.hidden_cell)
    #     hidden_out = F.elu(self.layer_hidden(self.hidden_cell[0][-1]))
    #     value_out = self.layer_value(hidden_out)
    #     return value_out


class ActorCritic(pl.LightningModule):
    def __init__(self, state_dim, action_dim, hidden_size, recurrent_layers, actor_lr, critic_lr):
        super().__init__()
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.shared = create_lstm(state_dim, hidden_size, recurrent_layers)

        self.actor = Actor(self.shared, action_dim, hidden_size)

        self.critic = nn.Sequential(
            self.shared,
            LambdaModule(),
            nn.Linear(hidden_size, 1)
        )

    @torch.no_grad()
    def __call__(self, state, **kwargs):
        pi, actions = self.actor(state)
        log_p = Actor.get_log_prob(pi, actions)
        value = self.critic(state)
        return pi, actions, log_p, value.squeeze()

    def get_log_prob(self, pi, actions: torch.Tensor) -> torch.Tensor:
        logp = self.actor.get_log_prob(pi, actions)
        return logp
