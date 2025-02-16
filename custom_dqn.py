import torch
from torch import nn
from pathlib import Path
from collections import deque
import random, datetime, os

import gymnasium as gym
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
import json
import numpy as np
import matplotlib.pyplot as plt
import time, datetime
from network.network import conv_mlp_net
import matplotlib.pyplot as plt
import matplotlib

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# def load_config():
#     conf_file = 'env/config.json'
#     config = json.load(open(conf_file, 'r'))
    
#     config = config['config1']
#     return config


# env.reset()
# next_state, reward, done, trunc, info = env.step(actions=0)

# print(f"{next_state.shape},\n {reward},\n {done},\n {info}")

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        pass


class MECNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # print(input_dim)
        c, h = input_dim
        self.block_num = 3
        self.online = self.__build_nn(c, output_dim)
        self.target = self.__build_nn(c, output_dim)
        self.target.load_state_dict(self.online.state_dict())

        for p in self.target.parameters():
            p.requires_grad = False
        
    def forward(self, input, model):
        # print(input)
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)


    def __build_nn(self, c, output_dim):
        return conv_mlp_net(conv_in=c, conv_ch=512, mlp_in=output_dim*512, mlp_ch=1024, out_ch=output_dim,block_num=self.block_num)
    

class MECAgent:
    def __init__(self, state_dim, action_dim, save_dir, checkpoint=""):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        # device = torch.device(
        #     "cuda" if torch.cuda.is_available() else
        #     "mps" if torch.backends.mps.is_available else
        #     "cpu"
        # )
        self.device = "cpu"
        self.net = MECNet(self.state_dim, self.action_dim).to(dtype=torch.float32)
        self.net = self.net.to(device=self.device)

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99975
        self.exploration_rate_min = 0.1
        self.curr_step = 0
        self.save_every = 1e3

        if checkpoint != "":
            checkpoint_path = "train_dqn/mec_net_latest.chkpt"  # Update this path as needed
            if Path(checkpoint_path).exists():
                self.net, self.exploration_rate = self.load_model(self.net, checkpoint_path, self.device)

    def load_model(model, checkpoint_path, device="cpu"):
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model weights
        model.load_state_dict(checkpoint["model"])
        
        # Load exploration rate
        exploration_rate = checkpoint.get("exploration_rate", 1.0)  # Default to 1.0 if not found

        print(f"✅ Loaded model from {checkpoint_path}")

        return model, exploration_rate

    def act(self, state):
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOITE
        else:
            state = state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            actions_values = self.net(state, model="online")
            # print(actions_values)
            action_idx = torch.argmax(actions_values, axis=1).item()
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.curr_step += 1
        return action_idx
    
    

class MECAgent(MECAgent):
    def __init__(self, state_dim, action_dim, save_dir, checkpoint=""):
        super().__init__(state_dim, action_dim, save_dir, checkpoint)
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000,device=torch.device("cpu")))
        self.batch_size = 32

    def cache(self, state, next_state, action, reward, done):
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])
        # print({"state": state, "next_state": next_state, "action": action, "reward": reward, "done": done})
        self.memory.add(TensorDict({"state": state, "next_state": next_state, "action": action, "reward": reward, "done": done}, batch_size=[]))

    def recall(self):
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()


class MECAgent(MECAgent):
    def __init__(self, state_dim, action_dim, save_dir, checkpoint=""):
        super().__init__(state_dim, action_dim, save_dir, checkpoint)
        self.burnin = 50  # min. experiences before training
        self.learn_every = 2  # no. of experiences between updates to Q_online
        self.sync_every = 10
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.gamma = 0.9


    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())
    
    def save(self):
        save_path = (
            self.save_dir / f"mec_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"MECNet saved to {save_path} at step {self.curr_step}")

    def td_estimate(self, state, action):
        # print(state)
        # print(action)
        current_Q = self.net(state, model="online")[np.arange(0, self.batch_size, dtype=np.float32), action]
    
        return current_Q
    
    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[np.arange(0, self.batch_size), best_action]
        # print(reward)
        # print(next_Q)
        return (reward + (1 - done.float() * self.gamma * next_Q)).float()

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
        
        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None
        
        if self.curr_step % self.learn_every != 0:
            return None, None
        
        
        state, next_state, action, reward, done = self.recall()

        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)

        loss = self.update_Q_online(td_est, td_tgt)
        # print(td_est, " -------------------- ")
        return (td_est.mean().item(), loss)


class MECLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

        self.plot_interval = 1000  # Update every 1e2 steps
        self.steps = []
        # Enable interactive mode
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 5))

    def plot_rewards(self):
        """Continuously update the reward plot without closing it"""
        self.ax.clear()  # Clear the previous frame
        self.ax.plot(self.steps, self.ep_rewards, label="Reward", color="blue")

        self.ax.set_xlabel("Steps")
        self.ax.set_ylabel("Reward")
        self.ax.set_title("Reward vs Steps")
        self.ax.legend()
        self.ax.grid(True)

        plt.draw()  # Redraw the plot
        plt.pause(0.1)  # Pause to allow the figure to update

    def log_step(self, reward, loss, q, step):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1




    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

       
        # self.plot_rewards()

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_lengths", "ep_avg_losses", "ep_avg_qs", "ep_rewards"]:
            plt.clf()
            plt.plot(getattr(self, f"moving_avg_{metric}"), label=f"moving_avg_{metric}")
            plt.legend()
            plt.savefig(getattr(self, f"{metric}_plot"))



