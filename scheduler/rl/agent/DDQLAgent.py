from .BaseAgent import BaseAgent
from network.net import SingleQNet
from buffer.per import PrioritizedReplayBuffer
import os
import sys
import torch
from pathlib import Path
import numpy as np

class DDQLAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, save_dir, checkpoint=""):
        super().__init__(state_dim, action_dim, save_dir, checkpoint)
        self.net = SingleQNet(self.state_dim, self.action_dim).to(dtype=torch.float32)

        if checkpoint != "":
            checkpoint_path = checkpoint
            if Path(checkpoint_path).exists():
                self.net, self.exploration_rate = self.load_model(self.net, checkpoint_path, self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0001)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.gamma = 0.2

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            actions_values = self.net(state, model="online")
            action_idx = torch.argmax(actions_values, axis=1).item()

        return action_idx
    
    def update_Q_online(self, td_estimate, td_target, IS_weights):
        loss = self.loss_fn(td_estimate, td_target) * IS_weights
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def td_estimate(self, states, actions):
        value_s, advantage_s = self.net(states, model="online")
        q_pred = torch.add(value_s, advantage_s - advantage_s.mean(dim=1, keepdim=True))[np.arange(self.batch_size), actions]
        return q_pred

    @torch.no_grad()
    def td_target(self, rewards, states, actions, next_states, dones):
        next_state_Q = self.net(next_states, model="online")
        best_actions = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_states,model="target")[np.arange(0, self.batch_size), best_actions]
        return (rewards + (1 - dones.float()) * self.gamma * next_Q).float()

    def learn(self):
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.curr_step += 1
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
        td_tgt = self.td_target(rewards, states, actions, next_states, dones)

        loss = self.update_Q_online(td_est, td_tgt, IS_weights)
        
        td_errors = td_est - td_tgt
        self.memory.update_priority(indices, td_errors.cpu().detach().numpy())
        return td_est.mean().item(), loss
    
    def predict(self, state):
        state = torch.tensor(state, device=self.device).unsqueeze(0)
        actions_values = self.net(state, model="online")
        action_idx = torch.argmax(actions_values, axis=1).item()
        return action_idx