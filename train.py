import os
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from .proximal_policy_optimization import RiskAwarePPO
from .envs.mock_cannoli_env import MockCannoliEnv
import yaml
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint

try:
    import gym
except  ModuleNotFoundError:
    _GYM_AVAILABLE = False
else:
    _GYM_AVAILABLE = True

CONFIG_DIR = f"{os.getcwd()}" + "/configuration/"


class Navigator:

    def __init__(self, exec_identifier, iteration_number):
        self.exec_identifier = exec_identifier

        if not _GYM_AVAILABLE:
            raise ModuleNotFoundError('This Module requires gym environment which is not installed yet.')

        self.x = None
        with open(CONFIG_DIR + self.exec_identifier + '.yaml', 'r') as file:
            self.x = yaml.safe_load(file)

        self.dir_path = os.getcwd() + "/" + f"previous_runs/iteration_{iteration_number}" + "/"
        print(self.dir_path)

        self.ppo = self.main()

        # this logs the best performing model weights every epoch (according to the val loss)
        checkpoint_callback = ModelCheckpoint(dirpath=self.dir_path, save_top_k=1, monitor="val_loss")

        self.trainer = Trainer(accelerator=self.x['accelerator'], devices=self.x['devices'],
                               max_epochs=self.x['epochs'], callbacks=[checkpoint_callback])

    def main(self):
        with open(CONFIG_DIR + self.x['actions'], 'r') as file:
            actions = yaml.safe_load(file)

        env = MockCannoliEnv(self.exec_identifier, self.x['steps'], np.array(actions))

        print(f" Value: {self.x['batch_size']}, Data type {type(self.x['batch_size'])}")

        return RiskAwarePPO(
            self.dir_path,
            env,
            self.x['batch_size'],
            self.x['episodes'],
            self.x['steps'],
            self.x['nb_optim_iters'],
            tuple(map(float, self.x["learning_rate"].strip("()").split(","))),
            self.x['value_loss_coef'],
            self.x['entropy_beta'],
            self.x['clip_ratio'],
            self.x['gamma'],
            self.x['lam'],
            self.x['risk_aware'],
            self.x['k'],
            self.x['curious'],
            self.x['state_latent_size'],
            self.x['policy_weight'],
            self.x['reward_scale'],
            self.x['weight'],
            self.x['intrinsic_reward_integration']
        )

    def navigate(self):
        self.trainer.fit(self.ppo)
