from workload.BitbrainWorkload import BWGD
from datacenter.BitbrainDC import BitbrainDC
import gymnasium as gym


class Env():
    def __init__(self):
        self.workload = BWGD(
            meanNumContainers=1,
            sigmaNumContainers=0.2
        )

        self.datacenter = BitbrainDC(num_hosts=5)

        self.action_space = None
        self.observation_spaces = None

        self.initialized_flag = False

    def reset(self, seed=42):
        if not self.initialized_flag:
            pass

    def step(self, action):
        pass

    def get_obs(self):
        pass

    def calc_rew(self):
        pass
