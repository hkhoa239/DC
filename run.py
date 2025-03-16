from scheduler.scheduler import Scheduler
import json
from utils.load_env import load_env
import random

def load_config(json_path,part):
    with open(json_path, 'r') as json_file:
        cfg = json.load(json_file)
    return cfg[f"config{part}"]

def train():
    config = load_config(json_path="config/env_config.json",part=1)
    env = load_env(params=config)
    sche = Scheduler(env=env, model="ddqn")
    sche.train()

def simulate():
    config = load_config(json_path="config/env_config.json",part=1)
    env = load_env(params=config)
    sche = Scheduler(env=env, model="ddqn", checkpoint_path="model/net_2.chkpt")
    sche.simulate()


def random_simulate():
    config = load_config(json_path="config/env_config.json",part=1)
    env = load_env(params=config)
    env.reset()
    while True:
        action = random.randint(0,10-1)
        state, reward, done, info = env.step(env.filter_action(9))
        print(reward)
        # step += 1

if __name__=="__main__":
    train()