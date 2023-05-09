import gym
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.optim as optim
from .reward.intrinsic_curiosity_module import ICM
from .agent.recurrent_actor_critic import ActorCritic
from typing import List
from .agent.replay_buffer import MiniBatch, Batch, Episode
import csv
from tdigest import TDigest


def discount_rewards(rewards: List[float], discount: float) -> list:
    # computes the discounted reward given the list of rewards and a discount coefficient

    cumul_reward = []
    sum_r = 0.0

    for r in reversed(rewards):
        sum_r = (sum_r * discount) + r
        cumul_reward.append(sum_r)

    r = list(reversed(cumul_reward))
    return r


def calc_advantage(rewards: list, values: list, last_value: float, gamma: float, lam: float) -> list:
    # generalized advantage estimation
    rews = rewards + [last_value]
    vals = values + [last_value]

    delta = [rews[i] + gamma * vals[i + 1] - vals[i] for i in range(len(rews) - 1)]

    return discount_rewards(delta, gamma * lam)


'''
proximal policy optimization 

for iteration i....t
    for actor:
        run old_policy
        compute advantage estimates 
    end
    optimization surrogate for K epochs consisting of B mini-batches of sampled data
    update new_policy with old_policy
'''


class RiskAwarePPO(pl.LightningModule):

    def __init__(
            self,
            path_to_specification: str,
            env,
            batch_size: int,
            episodes: int,
            steps_per_episode: int,
            nb_optim_steps: int,
            learning_rate: tuple,
            value_loss_coef: float,
            entropy_beta: float,
            clip_ratio: float,
            gamma: float,
            lam: float,
            risk_aware: bool,
            k: float,
            curious: bool,
            state_latent_size: int,
            policy_weight: float,
            reward_scale: float,
            weight: float,
            intrinsic_reward_integration: float
    ):
        super().__init__()

        self.dir = path_to_specification

        self.env = env
        self.obs_shape, self.action_shape = self.env.observation_space.shape, env.action_space.shape[0]

        print(f"Action shape {self.action_shape}")
        self.action_space = env.action_space
        self.obs_space = env.observation_space

        self.actor_lr, self.critic_lr = learning_rate
        self.agent = ActorCritic(state_dim=7, action_dim=self.action_shape,
                                 hidden_size=32, recurrent_layers=8, actor_lr=self.actor_lr, critic_lr=self.critic_lr)

        self.gamma = gamma
        self.lam = lam

        self.batch_size = batch_size
        self.episodes = episodes
        self.steps_per_episode = steps_per_episode
        self.nb_optim_steps = nb_optim_steps

        self.batch_size = batch_size

        self.automatic_optimization = True

        if risk_aware:
            self.k = k

        if curious:
            self.icm = ICM(self.obs_shape[0], self.action_shape, state_latent_size, policy_weight, reward_scale, weight)
            self.intrinsic_reward_integration = intrinsic_reward_integration

        self.value_loss_coef = value_loss_coef
        self.entropy_beta = entropy_beta
        self.clip_ratio = clip_ratio

        self.batch = MiniBatch()

        # encapsulates all the collections that compose an episode
        self.episode = Episode()

        self.epoch_rewards = []

        self.episode_step = 0
        self.avg_ep_reward = 0

        self.state = self.env.reset()

    def forward(self, x: torch.Tensor):
        pi, action = self.agent.actor(x)
        value = self.agent.critic(x)

        return pi, action, value

    def _compute_episode_reward(self, rewards, values):
        last_value = values[-1][-1]

        # enrich rewards with intrinsic reward
        # intr_temp_reward = self.icm.temp_reward(self.episode.rewards, self.episode.states, self.episode.actions)
        # intr_contr_reward = self.icm.oracle_reward(self.episode.rewards, self.episode.states, self.episode.actions)
        #
        # intrinsic = 0.5 * intr_temp_reward + 0.5 * intr_contr_reward
        # agg_rewards = (1. - self.intr_weight) * self.episode.rewards + self.intr_weight * intrinsic
        agg_rewards = rewards

        qvals = discount_rewards(agg_rewards + [last_value], self.gamma)[:-1]
        adv = calc_advantage(agg_rewards, values, last_value, self.gamma, self.lam)

        return qvals, adv

    def train_batch(self) -> tuple:
        for episode_idx in range(self.episodes):
            for step in range(self.steps_per_episode):

                pi, action, log_prob, value = self.agent(self.state)

                next_state, reward, done, _ = self.env.step(action.cpu().numpy())
                self.episode_step += 1
                # TODO: make it more clear state is actually the observation space
                self.batch.update_experience(state=self.state[step, :], next_state=next_state, action=action[step],
                                             logp=log_prob[step])
                self.episode.update(reward=reward, value=value, state=self.state, action=action)

                self.state = next_state

                terminal = len(self.episode.rewards) == self.steps_per_episode

                if done or terminal:
                    qvals, adv = self._compute_episode_reward(self.episode.rewards, self.episode.values)
                    self.epoch_rewards.append(sum(self.episode.rewards))
                    # reset episode
                    self.episode.reset()
                    self.episode_step = 0
                    self.state = self.env.reset()

                    # TODO: taking the last computed advantage and qvals (unclear exactly how to do this with
                    #  recurrence)

                    yield torch.stack(self.batch.states), torch.stack(self.batch.next_states), torch.stack(
                        self.batch.actions), \
                        torch.stack(self.batch.logp), adv[-1], qvals[-1]

                    self.batch.reset()

            self.avg_ep_reward = sum(self.epoch_rewards) / self.steps_per_episode

    def configure_optimizers(self) -> tuple:
        # initialize optimizer
        optimizer_actor = optim.Adam(self.agent.actor.parameters(), lr=self.actor_lr)
        optimizer_critic = optim.Adam(self.agent.critic.parameters(), lr=self.critic_lr)

        return optimizer_actor, optimizer_critic

    def optimizer_step(self, *args, **kwargs):
        for i in range(self.nb_optim_steps):
            super().optimizer_step(*args, **kwargs)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=Batch(self.train_batch), batch_size=self.batch_size)

    def training_step(self, batch: tuple, batch_idx, optimizer_idx):
        state, next_state, action, old_logp, adv, qval = batch

        # state = torch.squeeze(state, 2)
        # state = torch.squeeze(state, 2)
        #
        # old_logp, adv, qval = torch.unsqueeze(old_logp, 1), torch.unsqueeze(adv, 1), torch.unsqueeze(qval, 1)

        adv = (adv - adv.mean()) / adv.std()

        self.log("avg_ep_reward", self.avg_ep_reward, prog_bar=True, on_step=False, on_epoch=True)

        # TODO: this is where you should compute the risk aware policy cache
        td = TDigest()

        for q in range(qval.size(0)):
            e = qval[q]
            td.update(e)
            top_quintile = td.percentile(self.k)
            if e >= top_quintile:
                with open(self.dir + f'epoch_{len(self.epoch_rewards)}.csv', 'w', newline='') as csvfile:
                    reproducibility_criteria = csv.writer(csvfile)
                    reproducibility_criteria.writerow(list(map(lambda x: x, action)))

        if optimizer_idx == 0:
            loss_actor = self.actor_loss(state, action, old_logp, adv)
            self.log('loss_actor_raw', loss_actor, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            # intrinsic_loss = self.icm.loss(loss_actor, action, next_state, state)
            # self.log('loss_actor_curious', loss_actor, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            return loss_actor

        elif optimizer_idx == 1:
            loss_critic = self.critic_loss(state, qval)
            self.log('loss_critic', loss_critic, on_step=False, on_epoch=True, prog_bar=False, logger=True)

            return loss_critic

    def get_device(self, batch) -> str:
        return batch[0].device.index if self.on_gpu else 'cpu'

    def actor_loss(self, state, action, logp_old, adv) -> torch.Tensor:
        pi, _ = self.agent.actor(state)
        logp = self.agent.actor.get_log_prob(pi, action)
        ratio = torch.exp(logp - logp_old)

        # this is the PPO bit - i.e. pessimistic update of policy minimizing amount of entropy epoch over epoch
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_actor = -(torch.min(ratio * adv, clip_adv)).mean()
        return loss_actor

    def critic_loss(self, state, qval) -> torch.Tensor:
        value = self.agent.critic(state)
        loss_critic = (qval - value).pow(2).mean()
        return loss_critic
