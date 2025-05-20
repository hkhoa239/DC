import torch
import torch.optim as opt
import torch.nn.functional as F
import numpy as np
from pathlib import Path 

from ..network.PPOActorCriticNet import PPOActorCriticNet

class PPOAgent():
    def __init__(self, state_dim, action_dim, lr_actor_critic, 
                 gamma, ppo_epochs, clip_epsilon, gae_lambda, 
                 entropy_coef, vf_coef, device, save_dir=None, 
                 checkpoint_path_actor_critic=None):
        
        self.device = torch.device(device if torch.cuda.is_available() and device=="cuda" else "cpu")
        print(f"PPO Agent using device: {self.device}")

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Hyperparameters
        self.gamma = gamma
        self.ppo_epochs = ppo_epochs
        self.clip_epsilon = clip_epsilon
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef

        # Networks
        self.actor_critic_net = PPOActorCriticNet(state_dim, action_dim).to(self.device)
        
        # Optimizer 
        self.optimizer = opt.Adam(self.actor_critic_net.parameters(), lr = lr_actor_critic)

         # on-policy memory
        self.memory_states = []
        self.memory_actions = []
        self.memory_log_probs = []
        self.memory_rewards = []
        self.memory_values = [] # V(s_t) từ critic lúc thu thập
        self.memory_dones = []

        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        self.curr_step_count = 0

        if checkpoint_path_actor_critic and Path(checkpoint_path_actor_critic).exists():
            self._load_network(self.actor_critic_net, checkpoint_path_actor_critic, self.optimizer, "actor_critic")
    
    def _store_transition_tensor(self, state_t, action_t, log_prob_t, value_t, reward_t, done_t):
        """Lưu trữ các tensor đã được chuyển lên device."""
        self.memory_states.append(state_t)         # state đã là tensor
        self.memory_actions.append(action_t)       # action đã là tensor
        self.memory_log_probs.append(log_prob_t)   # log_prob đã là tensor
        self.memory_rewards.append(reward_t)       # reward đã là tensor
        self.memory_values.append(value_t)         # value đã là tensor
        self.memory_dones.append(done_t)           # done đã là tensor

    def clear_memory(self):
        self.memory_states = []
        self.memory_actions = []
        self.memory_log_probs = []
        self.memory_rewards = []
        self.memory_values = []
        self.memory_dones = []

    @torch.no_grad()
    def select_action(self, state_np: np.ndarray):
        """Chọn action từ state (numpy array) hiện tại."""
        state_tensor = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        self.actor_critic_net.eval()

        action_dist = self.actor_critic_net.get_action_distribution(state_tensor)
        action = action_dist.sample() 
        log_prob_action = action_dist.log_prob(action)
        value_estimate = self.actor_critic_net.get_value(state_tensor)

        # Không cần chuyển actor/critic về train() ở đây, vì learn() sẽ làm.
        # Trả về action (int), và các tensor log_prob, value
        return action.item(), log_prob_action, value_estimate # log_prob, value là tensor [1] hoặc [1,1]
    
    def _calculate_advantages_gae(self, last_value_tensor: torch.Tensor):
        # Chuyển list các tensor đơn lẻ thành một batch tensor
        rewards_batch = torch.cat(self.memory_rewards) # Shape: (N,)
        values_batch = torch.cat(self.memory_values).squeeze(-1)   # Shape: (N,)
        dones_batch = torch.cat(self.memory_dones)      # Shape: (N,)
        
        num_steps = len(rewards_batch)
        advantages = torch.zeros(num_steps, device=self.device)
        returns = torch.zeros(num_steps, device=self.device)
        
        gae = 0.0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - dones_batch[t].float()
                next_value = last_value_tensor.squeeze() * next_non_terminal # last_value_tensor có thể là [1] hoặc [1,1]
            else:
                next_non_terminal = 1.0 - dones_batch[t].float()
                next_value = values_batch[t+1] * next_non_terminal
            
            delta = rewards_batch[t] + self.gamma * next_value - values_batch[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = gae + values_batch[t] # Target cho value function
            
        return advantages, returns
    
    def learn(self, last_value_np: np.ndarray, last_done: bool):
        """
        Học từ batch dữ liệu đã thu thập.
        last_value_np: V(s_T+1) của state cuối cùng (numpy array), nếu không done.
        last_done: Done flag của state cuối cùng.
        """
        if not self.memory_states: # Không có gì để học
            return None, None

        # Tính last_value_tensor cho GAE
        if last_done:
            last_value_tensor = torch.tensor([0.0], dtype=torch.float32, device=self.device)
        else:
            # Cần critic đánh giá state cuối cùng này
            # Hàm này phải được gọi TRƯỚC khi clear_memory
            self.critic.eval() # Đảm bảo critic ở eval mode
            with torch.no_grad():
                last_state_tensor = torch.tensor(last_value_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                last_value_tensor = self.critic(last_state_tensor) # Shape [1,1]
            # self.critic.train() # Chuyển lại train sau, hoặc để learn() tự xử lý

        advantages, returns = self._calculate_advantages_gae(last_value_tensor)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Chuyển list các tensor thành batch tensor
        old_states_batch = torch.cat(self.memory_states)       # Shape: (N, state_dim)
        old_actions_batch = torch.cat(self.memory_actions)     # Shape: (N,)
        old_log_probs_batch = torch.cat(self.memory_log_probs).squeeze(-1) # Shape: (N,)
                                                                    # Squeeze nếu log_probs là [N,1]

        actor_losses, critic_losses, entropy_bonuses = [], [], []

        for _ in range(self.ppo_epochs):
            # Tạo mini-batches nếu cần, ở đây dùng cả batch
            
            # --- Actor Loss ---
            self.actor_critic_net.train()
            action_dist_new = self.actor_critic_net.get_action_distribution(old_states_batch)
            new_log_probs = action_dist_new.log_prob(old_actions_batch) # old_actions_batch là index
            entropy = action_dist_new.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs_batch) # old_log_probs_batch không có grad
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            actor_total_loss = policy_loss - self.entropy_coef * entropy # Trừ entropy bonus (vì muốn maximize entropy)

            self.optimizer.zero_grad()
            actor_total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5) # Optional: gradient clipping
            self.optimizer.step()

            # --- Critic Loss ---
            self.actor_critic_net.train()
            # new_values là V(s_t) được ước lượng bởi critic *hiện tại* cho các state trong batch
            new_values = self.critic(old_states_batch).squeeze(-1) # Shape: (N,)
            value_loss = F.mse_loss(new_values, returns) # returns là target V_target = GAE_t + V_old(s_t)

            self.optimizer.zero_grad()
            critic_total_loss = self.vf_coef * value_loss
            critic_total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5) # Optional
            self.optimizer.step()

            actor_losses.append(policy_loss.item()) # Chỉ lưu policy loss (chưa trừ entropy)
            critic_losses.append(value_loss.item())
            entropy_bonuses.append(entropy.item())

        self.clear_memory()
        
        return np.mean(actor_losses), np.mean(critic_losses), np.mean(entropy_bonuses)