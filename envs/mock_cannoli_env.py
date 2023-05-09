# how to map from a sampled action to a concrete high-level instruction
import gym

import numpy as np
import simplejson as json
import torch

with open('/Users/laurenmoos/leetasm-private/navigator/envs/sample_json/response-cromulence.json') as file:
    heap_state = json.loads(file.read())


def _generate_faux_response(pid, seed):
    return heap_state


class CannoliMockClient:
    # encapsulates a unique socket connection to cannoli identified with a unique PID

    def __init__(self, pid):
        self.pid = pid

    def streaming(self, action) -> dict:

        return _generate_faux_response(self.pid, action)

    def disconnect_session(self):
        pass


class MockCannoliEnv(gym.Env):
    def render(self, mode="human"):
        pass

    def __init__(self, exec_identifier: str, max_steps_episode: int, action_space: np.array):
        super(MockCannoliEnv, self).__init__()
        self.client: CannoliMockClient = CannoliMockClient(1223)

        self.last_state = self.client.streaming(11)
        # TODO: once checking against a specification better to keep this as an AST
        self.observation_shape = (max_steps_episode, 7)
        self.observation_space = torch.zeros(self.observation_shape)
        self.action_space = action_space
        self.episode_reward = 1.0
        self.steps_left = max_steps_episode

    def distance(self, oracle_next_state, next_state):
        return

    def step(self, action):
        episode_terminated = False

        response = self.client.streaming(action)

        ini = float(response['ini'])
        lockpin = float(response['lockpin'])
        p_r = float(response['p_r'])
        p_w = float(response['p_w'])
        lockpin_crc = float(response["lockpin_crc"])
        req_crc = float(response["req_crc"] or 0)
        ret = float(response["return"] or 0)

        #TODO: this is now a place holder
        entropy, memory = 1, 1
        next_state = torch.tensor([ini, lockpin, p_r, p_w, lockpin_crc, req_crc, ret])
        self.episode_reward += entropy * memory

        self.steps_left -= 1
        if not self.steps_left:
            episode_terminated = True

        self.last_state = next_state
        torch.cat((self.observation_space, next_state.unsqueeze(0)), dim=0)

        return self.observation_space, self.episode_reward, episode_terminated, []

    def reset(self):

        self.client.disconnect_session()
        self.episode_reward = 0
        self.last_state = None

        return torch.zeros(self.observation_shape)
