from scheduler.scheduler import Scheduler
import json
from utils.load_env import load_env
def load_config(json_path,part):
    with open(json_path, 'r') as json_file:
        cfg = json.load(json_file)
    return cfg[f"config{part}"]

def main():
    config = load_config(json_path="config/env_config.json",part=1)
    env = load_env(params=config)
    sche = Scheduler(env=env, model="ddqn")
    sche.train()

if __name__=="__main__":
    main()