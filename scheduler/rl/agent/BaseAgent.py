import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import torch
from torch import nn
from pathlib import Path
from buffer.per import PrioritizedReplayBuffer

class BaseAgent:
    def __init__(self, state_dim, action_dim, save_dir, checkpoint=""):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.device = "cpu"
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0
        self.save_every = 5000
        self.burnin = 100
        self.learn_every = 2
        self.sync_every = 500

        self.memory = PrioritizedReplayBuffer(capacity=100000)  # Use PER buffer
        self.batch_size = 32

    def save(self):
        save_path = self.save_dir / f"net_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate), save_path)
        print(f"Net saved to {save_path} at step {self.curr_step}")

    def load_model(self, model, checkpoint_path, device="cpu"):
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        exploration_rate = checkpoint.get("exploration_rate", 1.0)
        print(f"✅ Loaded model from {checkpoint_path}")

        return model, exploration_rate
    
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
    
    
    