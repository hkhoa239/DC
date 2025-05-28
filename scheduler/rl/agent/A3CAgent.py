from .BaseAgent import BaseAgent
from network.net import ActorCritic
from buffer.per import PrioritizedReplayBuffer
import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def train(rank, args, shared_model, counter, lock, env, optimizer=None):
    torch.manual_seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape, env.observation_space.shape).to(dtype=torch.float32)

    if optimizer is None:
        optimizer = torch.optim.Adam(shared_model.parameters(), lr=args.lr)
    
    model.train()
    state, _ = env.reset()
    state = torch.from_numpy(state)

    done = True
    episode_length = 0
    while True:
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = torch.zeros(1,256)
            hx = torch.zeros(1,256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []
        for step in range(args.num_steps):
            episode_length += 1
            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)

            state, reward, done, _ = env.step(action.numpy())
            done = done or episode_length >= args.max_episode_length
            reward = max(min(reward, 1), -1)

            with lock:
                counter.value += 1
            if done:
                episode_length = 0
                state, _ = env.reset()
            
            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break
        
        R = torch.zeros(1,1)
        if not done:
            value, _, _ = model((state.unsqueeze(0), (hx, cx)))
            R = value.detach()
        
        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
    
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            delta_t = rewards[i] + args.gamma * values[i + 1] - values[i]
            gae = gae * args.gamma * args.gae_lambda + delta_t

            policy_loss = policy_loss - log_probs[i] * gae.detach() - args.entropy_coef * entropies[i]

        optimizer.zero_grad()

        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()

    

class A3CAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, save_dir, checkpoint=""):
        super().__init__(state_dim, action_dim, save_dir, checkpoint)
        self.net = ActorCritic(self.state_dim, self.action_dim).to(dtype=torch.float32)
        self.net.share_memory()

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
        current_Q = self.net(states, model="online")[np.arange(0, self.batch_size), actions]
        return current_Q

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
    
    def predict(self, state, state_pair):
        state = torch.tensor(state, device=self.device).unsqueeze(0)
        actions_values = self.net((state, state_pair), model="online")[1]
        action_idx = torch.argmax(actions_values, axis=1).item()
        return action_idx % 21