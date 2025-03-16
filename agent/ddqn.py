import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch
from torch import nn
from pathlib import Path
from collections import deque
import random, datetime, os
import torch.nn.functional as F
import gymnasium as gym
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
import json
import numpy as np
import matplotlib.pyplot as plt
import time, datetime
import matplotlib.pyplot as plt
import matplotlib
from buffer.per import PrioritizedReplayBuffer
from envs.Env import Env
from network.net import Net, DuelingQNet



class DDQNAgent:
    def __init__(self, state_dim, action_dim, save_dir, checkpoint=""):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.device = "cpu"
        self.net = DuelingQNet(self.state_dim, self.action_dim).to(dtype=torch.float32)
        self.net = self.net.to(device=self.device)

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99975
        self.exploration_rate_min = 0.1
        self.curr_step = 0
        self.save_every = 500

        self.memory = PrioritizedReplayBuffer(capacity=100000)  # Use PER buffer
        self.batch_size = 32

        self.burnin = 100  # min. experiences before training
        self.learn_every = 2  # no. of experiences between updates to Q_online
        self.sync_every = 200
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0001)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.gamma = 0.9

        if checkpoint != "":
            checkpoint_path = checkpoint
            if Path(checkpoint_path).exists():
                self.net, self.exploration_rate = self.load_model(self.net, checkpoint_path, self.device)

    def load_model(self, model, checkpoint_path, device="cpu"):
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        exploration_rate = checkpoint.get("exploration_rate", 1.0)
        print(f"✅ Loaded model from {checkpoint_path}")

        return model, exploration_rate

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            actions_values = self.net(state, model="online")
            action_idx = torch.argmax(actions_values, axis=1).item()
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        experience = (state, next_state, action, reward, done)
        self.memory.store(experience)

    def recall(self):
        batch, indices, IS_weights = self.memory.sample(self.batch_size)
        states, next_states, actions, rewards, dones = zip(*batch)
        states = torch.tensor(states).to(self.device)
        next_states = torch.tensor(next_states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        dones = torch.tensor(dones).to(self.device)
        IS_weights = torch.tensor(IS_weights).to(self.device)

        return states, next_states, actions, rewards, dones, indices, IS_weights

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[np.arange(0, self.batch_size), action]
        return current_Q

    def value_rescale(self, x, epsilon=1e-3):
        return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.0) - 1.0) + epsilon * x

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float() * self.gamma * next_Q)).float()

    @torch.no_grad()
    def td_target_munchausen(self, reward, state, action, next_state, done):
        tau = 0.03
        alpha = 0.9
        min_log_policy = -5.0

        # Q(s, a) from online net (used to compute Munchausen bonus)
        current_Q = self.net(state, model="online")
        log_pi = F.log_softmax(current_Q / tau, dim=1)
        munchausen_log_policy = log_pi[np.arange(0, self.batch_size), action]
        munchausen_term = alpha * torch.clamp(munchausen_log_policy, min=min_log_policy)

        # Double DQN: get best action from Q_online, evaluate from Q_target
        next_Q_online = self.net(next_state, model="online")
        best_action = torch.argmax(next_Q_online, axis=1)

        next_Q_target = self.net(next_state, model="target")
        next_Q = next_Q_target[np.arange(0, self.batch_size), best_action]

        # Final Munchausen target
        td_target = reward + munchausen_term + (1 - done.float()) * self.gamma * next_Q

        # Optional: Rescale target
        td_target = self.value_rescale(td_target)

        return td_target

    def update_Q_online(self, td_estimate, td_target, IS_weights):
        loss = (td_estimate - td_target).pow(2) * IS_weights
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        states, next_states, actions, rewards, dones, indices, IS_weights = self.recall()

        td_est = self.td_estimate(states, actions)
        td_tgt = self.td_target_munchausen(rewards, states, actions, next_states, dones)

        loss = self.update_Q_online(td_est, td_tgt, IS_weights)

        td_errors = td_est - td_tgt
        self.memory.update_priority(indices, td_errors.cpu().detach().numpy())

        return (td_est.mean().item(), loss)

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = self.save_dir / f"net_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate), save_path)
        print(f"Net saved to {save_path} at step {self.curr_step}")

    def predict(self, state):
        state = torch.tensor(state, device=self.device).unsqueeze(0)
        actions_values = self.net(state, model="online")
        # topk = torch.topk(actions_values, k=3, dim=1)  # returns values and indices
        # top3_indices = topk.indices.squeeze(0).tolist()  # list of top-2 indices
        # return top3_indices
        action_idx = torch.argmax(actions_values, axis=1).item()
        return action_idx