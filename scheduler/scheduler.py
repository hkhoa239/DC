import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from pathlib import Path
from scheduler.log import Logger
from agent.ddqn import DDQNAgent
import datetime
import random


class Scheduler():
    def __init__(self, env, model="ddqn", episodes=10, checkpoint_path=""):
        self.env = env
        self.model = model
        self.save_dir = Path(f"train_{self.model}") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.checkpoint_path = ""
        if model == "ddqn":
            self.agent = DDQNAgent(state_dim=env.observation_space.shape, action_dim=env.action_space.n, save_dir=self.save_dir, checkpoint=self.checkpoint_path)

    def train(self):
        self.save_dir.mkdir(parents=True)
        
        checkpoint_path = ""
        logger = Logger(self.save_dir)
        

        episodes = 500

        state, info = self.env.reset()
        for e in range(episodes+1):
            # step = 0
            # while True:
            action = self.agent.act(state)
            decision = self.env.filter_action(action)
            next_state, reward, done, info = self.env.step(action=decision)
            # print(reward)
            
            # next_state, reward, done, info = env.step(action)
            self.agent.cache(state, next_state, action, reward, done)

            q, loss = self.agent.learn()

            logger.log_step(reward, loss, q, e)
            
            # print(step, "------------------", reward)

            # step += 1

            if done or e % 100 == 0:
              logger.log_episode()
              logger.record(episode=e, epsilon=self.agent.exploration_rate, step=self.agent.curr_step)
        self.env.initialized_flag = False

    def simulate(self):
        state, info = self.env.reset()
        step = 0
        while (step < 1000):
            action = self.agent.predict(state=state)
            print(action)
            state, reward, done, info = self.env.step(self.env.filter_action(action))
            print(reward)
            step += 1