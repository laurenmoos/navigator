import gym

import numpy as np
import torch
from .cannoli_streaming_client import CannoliStreamingClient


class CannoliEnv(gym.Env):
    """
    So once bench.py spawns the processes and CannoliStreamingClient (NAV side of socket) connects with Middleware (tracer side of socket), I believe the following operations happen for a step.
    1. CannoliEnv steps with step https://github.com/narfindustries/leetasm-private/blob/low-level-integration/navigator/envs/cannoli_env.py#L34-L55
    2. Step calls _atomic_transaction  which calls CannoliStreamingClient.try_write() with a command for the tracer. Commands for this target should be in tuples like we previously discussed. This writes data over the socket to Middleware: https://github.com/narfindustries/leetasm-private/blob/low-level-integration/navigator/envs/cannoli_streaming_client.py#L65-L74. After calling try_write(), CannoliEnv calls CannoliStreamingClient.try_read() which will wait for feedback from the tracer
    3. Middleware receives the command and proxies it to the tracer: https://github.com/narfindustries/leetasm-uefi-testbench/blob/nav-integration/gdblib/middleware.py#L25-L30
    4. The tracer handles the command and hits callbacks. This is all in the guts of the tracer. eventually, it reaches the end of the command and serializes all the events into JSON. It passes that JSON to Middleware which packages it up and sends to CannoliStreamingClient: https://github.com/narfindustries/leetasm-uefi-testbench/blob/nav-integration/gdblib/middleware.py#L56-L61
    5. CannoliStreamingClient receives and deserializes the events. It also performs invariant checking and memoization here. Right now that is stubbed out in a check function, which will later be replaced by Aaron's TLA+ code. At the end of the function, it should return whatever relevant data NAV needs. Right now this is nothing, I can change very easily once we know what NAV wants: https://github.com/narfindustries/leetasm-private/blob/low-level-integration/navigator/envs/cannoli_streaming_client.py#L98-L121
    6. This return ends atomic_transaction and CannoliEnv does the rest of the step code
    Rinse and repeat until end of episode (in which client.reset() is called)
    """
    def render(self, mode="human"):
        pass

    def __init__(self, executable_identifier, max_steps_episode: int, action_space: np.array):
        super(CannoliEnv, self).__init__()
        self.max_steps_episode = max_steps_episode
        self.executable_identifier = executable_identifier
        self.client: CannoliStreamingClient = CannoliStreamingClient(executable_identifier)
        self.last_state = None

        # TODO: once checking against a specification better to keep this as an AST
        self.observation_shape = (max_steps_episode, 7)
        self.observation_space = torch.zeros(self.observation_shape)
        self.action_space = action_space
        self.episode_reward = 1.0
        self.steps_left = max_steps_episode

    def distance(self, oracle_next_state, next_state):
        return

    def _atomic_transaction(self, action):
        self.client.try_write("1")
        return self.client.try_read()

    def step(self, action):
        episode_terminated = False
        action = str(action.item())
        response = self._atomic_transaction(action)

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
        self.client = CannoliStreamingClient(self.executable_identifier)
        self.observation_space = []
        self.episode_reward = 0
        self.last_state = None
        self.steps_left = self.max_steps_episode

        return torch.zeros(self.observation_shape)

