import torch
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from pathlib import Path
# import matplotlib.pyplot as plt

import sys
import os

# Lấy đường dẫn đến thư mục chứa file PPO.py (tức là 'scheduler')
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Lấy đường dẫn đến thư mục gốc của dự án ('DC')
project_root = os.path.dirname(current_script_dir)

# Thêm thư mục gốc của dự án vào sys.path nếu chưa có
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from envs.Env import Env 
from scheduler.rl.agent.PPOAgent import PPOAgent 

VM_FEATURE_DIM = 5  # CPU, Core, RAM, BW Up, BW Down from VM

# ----- HYPERPARAMETERS (Có thể được truyền vào hàm train) -----
# DEFAULT_ENV_PARAMS = {
#     "TotalPower": 2000,
#     "RouterBw": 10000,
#     "NumHost": 10,
#     "VmLimit": 50,
#     "IntervalTime": 1,
#     "max_steps_per_episode": 200 # Số LẦN THỬ ĐẶT VM trong 1 PPO episode
# }

# DEFAULT_PPO_AGENT_PARAMS = {
#     "lr_actor_critic": 3e-4,
#     "gamma": 0.99,
#     "ppo_epochs": 10,
#     "clip_epsilon": 0.2,
#     "gae_lambda": 0.95,
#     "entropy_coef": 0.01,
#     "vf_coef": 0.5,
#     "device": "cuda" if torch.cuda.is_available() else "cpu",
#     "save_dir": "ppo_checkpoints_direct_env_refactored",
#     "checkpoint_path_actor_critic": None
# }

# DEFAULT_TRAINING_PARAMS = {
#     "total_training_steps": 1_000_000,
#     "steps_per_batch_collection": 2048,
#     "save_model_every_n_updates": 50,
#     "log_every_n_updates": 10
# }

TEST_ENV_PARAMS = {
    "TotalPower": 2000,
    "RouterBw": 10000,
    "NumHost": 5,  # Giảm số lượng host
    "VmLimit": 10, # Giảm giới hạn VM
    "IntervalTime": 1,
    "max_steps_per_episode": 50 # Giảm số bước thử đặt VM trong 1 episode
}

TEST_PPO_AGENT_PARAMS = {
    "lr_actor_critic": 3e-4,
    "gamma": 0.99,
    "ppo_epochs": 4, # Giảm số epoch học trên 1 batch
    "clip_epsilon": 0.2,
    "gae_lambda": 0.95,
    "entropy_coef": 0.01,
    "vf_coef": 0.5,
    "device": "cpu", # Chắc chắn chạy trên CPU
    "save_dir": "ppo_checkpoints_test_run",
    "checkpoint_path_actor_critic": None
}

TEST_TRAINING_PARAMS = {
    "total_training_steps": 1000,  # Giảm tổng số bước huấn luyện
    "steps_per_batch_collection": 128, # Giảm số bước thu thập cho mỗi batch
    "save_model_every_n_updates": 10, # Lưu model thường xuyên hơn (nếu cần)
    "log_every_n_updates": 5 # Log thường xuyên hơn
}


def get_vm_features(vm_instance):
    if vm_instance:
        return np.array([
            vm_instance.cpu,
            vm_instance.core,
            vm_instance.ram,
            vm_instance.bwup,
            vm_instance.bwdown
        ], dtype=np.float32)
    return np.zeros(VM_FEATURE_DIM, dtype=np.float32)

def create_ppo_specific_state(original_host_state_np, vm_features_np, num_hosts, expected_host_features_per_host=4):
    """
    Tạo state mở rộng cho PPO từ state host gốc và features của VM.
    original_host_state_np: state trả về từ env.getState().
    vm_features_np: numpy array chứa features của VM được chọn.
    num_hosts: Số lượng host, để xác định kích thước flat_host_features.
    expected_host_features_per_host: Số features trên mỗi host trong state gốc (ví dụ: 4).
    """
    expected_host_flat_dim = num_hosts * expected_host_features_per_host

    if len(original_host_state_np.shape) > 1:
        # Kiểm tra shape gốc và flatten cho đúng
        # Original env.observation_space.shape is (4, NumHost)
        if original_host_state_np.shape == (expected_host_features_per_host, num_hosts):
            flat_host_features = original_host_state_np.flatten()
        elif original_host_state_np.shape == (num_hosts, expected_host_features_per_host):
             # Nếu monitor.hostMatrix là (NumHost, 4) thì cần flatten() là đủ
            flat_host_features = original_host_state_np.flatten()
        else:
            raise ValueError(f"Unexpected original_host_state_np shape: {original_host_state_np.shape}. "
                             f"Expected ({expected_host_features_per_host}, {num_hosts}) or ({num_hosts}, {expected_host_features_per_host}).")
    else: # Đã là 1D
        flat_host_features = original_host_state_np
    
    if len(flat_host_features) != expected_host_flat_dim:
        raise ValueError(f"Dimension mismatch for flattened host state. "
                         f"Expected {expected_host_flat_dim}, got {len(flat_host_features)} "
                         f"from original shape {original_host_state_np.shape}")

    combined_state = np.concatenate((flat_host_features, vm_features_np)).astype(np.float32)
    return combined_state

# def plot_training_results(log_data, save_dir):
#     """Vẽ đồ thị kết quả huấn luyện."""
#     episode_rewards_log = log_data.get("rewards", [])
#     actor_losses_log = log_data.get("actor_losses", [])
#     critic_losses_log = log_data.get("critic_losses", [])
#     entropy_log = log_data.get("entropy", [])

#     if not any([episode_rewards_log, actor_losses_log, critic_losses_log, entropy_log]):
#         print("No data to plot.")
#         return

#     save_plot_dir = Path(save_dir or ".")
#     save_plot_dir.mkdir(parents=True, exist_ok=True)

#     plt.figure(figsize=(18, 6))
#     if episode_rewards_log:
#         plt.subplot(1, 3, 1); plt.plot(episode_rewards_log); plt.title("Avg Reward per Batch"); plt.xlabel("Update");
    
#     if actor_losses_log or critic_losses_log:
#         plt.subplot(1, 3, 2);
#         if actor_losses_log: plt.plot(actor_losses_log, label="Actor Loss")
#         if critic_losses_log: plt.plot(critic_losses_log, label="Critic Loss")
#         plt.title("Losses"); plt.xlabel("Update"); plt.legend()
    
#     if entropy_log:
#         plt.subplot(1, 3, 3); plt.plot(entropy_log); plt.title("Policy Entropy"); plt.xlabel("Update");
    
#     plt.tight_layout()
#     plt.savefig(save_plot_dir / "training_curves_direct_env_refactored.png")
#     print(f"Training curves saved to {save_plot_dir / 'training_curves_direct_env_refactored.png'}")


# ----- TRAIN FUNCTION -----
def train(env_params, agent_params, training_params):
    print("Starting PPO Training...")
    print(f"Environment Params: {env_params}")
    print(f"Agent Params: {agent_params}")
    print(f"Training Params: {training_params}")

    current_env_params = env_params.copy()
    max_steps_per_ppo_episode = current_env_params.pop("max_steps_per_episode", 200) 
    env = Env(**current_env_params) # Env gốc
    num_hosts = env_params["NumHost"]

    original_host_state_shape = env.observation_space.shape 
    expected_host_features_per_host = original_host_state_shape[0] 
    if not (len(original_host_state_shape) == 2 and \
            original_host_state_shape[0] == expected_host_features_per_host and \
            original_host_state_shape[1] == num_hosts):
        print(f"Warning: Original env.observation_space.shape is {original_host_state_shape}, "
              f"but expected ({expected_host_features_per_host}, {num_hosts}). Adjust flattening logic if needed.")
    
    original_flat_host_state_dim = original_host_state_shape[0] * original_host_state_shape[1]
    
    ppo_state_dim = original_flat_host_state_dim + VM_FEATURE_DIM
    action_dim = env.action_space.n # Discrete(NumHost)

    print(f"PPO specific state_dim: {ppo_state_dim} (Host_flat: {original_flat_host_state_dim} + VM: {VM_FEATURE_DIM})")
    print(f"Action dimension (NumHost): {action_dim}")

    agent = PPOAgent(state_dim=ppo_state_dim, action_dim=action_dim, **agent_params)

    log_data = {"rewards": [], "actor_losses": [], "critic_losses": [], "entropy": []}
    
    global_step_count = 0
    update_count = 0
    
    total_training_steps = training_params["total_training_steps"]
    steps_per_batch_collection = training_params["steps_per_batch_collection"]
    save_model_every_n_updates = training_params["save_model_every_n_updates"]
    log_every_n_updates = training_params["log_every_n_updates"]

    progress_bar = tqdm(total=total_training_steps, desc="Training Progress")
    
    while global_step_count < total_training_steps:
        num_steps_collected_this_batch = 0
        rewards_for_current_batch_episodes = []

        while num_steps_collected_this_batch < steps_per_batch_collection:
            original_host_state_np, info = env.reset() #
            current_ppo_episode_reward_sum = 0
            
            for ppo_step_in_episode in range(max_steps_per_ppo_episode):
                if global_step_count >= total_training_steps or \
                   num_steps_collected_this_batch >= steps_per_batch_collection:
                    break

                selectable_vms_ids = env.getSelectableVms() #
                if not selectable_vms_ids:
                    break 

                selected_vid = selectable_vms_ids[0]
                selected_vm_instance = env.getVmByID(selected_vid) #
                vm_feats_np = get_vm_features(selected_vm_instance)
                
                current_ppo_specific_state_np = create_ppo_specific_state(
                    original_host_state_np, vm_feats_np, num_hosts, expected_host_features_per_host
                )
                
                action_host_idx, log_prob_action_tensor, value_estimate_tensor = \
                    agent.select_action(current_ppo_specific_state_np)
                
                decision = [(selected_vid, action_host_idx)]
                state_for_memory_tensor = torch.tensor(current_ppo_specific_state_np, dtype=torch.float32, device=agent.device)

                next_original_host_state_np, reward, done_env, truncated_env, info = env.step(decision) #

                is_ppo_transition_terminal = done_env or truncated_env
                if ppo_step_in_episode == max_steps_per_ppo_episode - 1:
                    is_ppo_transition_terminal = True
                
                agent._store_transition_tensor(
                    state_for_memory_tensor,
                    torch.tensor([action_host_idx], dtype=torch.long, device=agent.device),
                    log_prob_action_tensor, value_estimate_tensor,
                    torch.tensor([reward], dtype=torch.float32, device=agent.device),
                    torch.tensor([is_ppo_transition_terminal], dtype=torch.bool, device=agent.device)
                )
                
                original_host_state_np = next_original_host_state_np
                current_ppo_episode_reward_sum += reward
                global_step_count += 1
                num_steps_collected_this_batch += 1
                progress_bar.update(1)

                if done_env or truncated_env:
                    break
            
            rewards_for_current_batch_episodes.append(current_ppo_episode_reward_sum)
            if global_step_count >= total_training_steps or \
               num_steps_collected_this_batch >= steps_per_batch_collection:
                break
        
        if num_steps_collected_this_batch > 0 and len(agent.memory_states) > 0:
            final_transition_in_batch_is_terminal = agent.memory_dones[-1].item()
            next_ppo_state_for_gae_value_np = None
            
            if not final_transition_in_batch_is_terminal:
                next_selectable_vms_ids = env.getSelectableVms() #
                if next_selectable_vms_ids:
                    next_vm_id_for_gae = next_selectable_vms_ids[0]
                    next_vm_instance_for_gae = env.getVmByID(next_vm_id_for_gae) #
                    next_vm_feats_np = get_vm_features(next_vm_instance_for_gae)
                    next_ppo_state_for_gae_value_np = create_ppo_specific_state(
                        original_host_state_np, next_vm_feats_np, num_hosts, expected_host_features_per_host
                    )
            
            effective_last_done_for_gae = final_transition_in_batch_is_terminal
            if not final_transition_in_batch_is_terminal and next_ppo_state_for_gae_value_np is None:
                effective_last_done_for_gae = True

            actor_loss_val, critic_loss_val, entropy_val = agent.learn(
                last_value_np=next_ppo_state_for_gae_value_np,
                last_done=effective_last_done_for_gae
            )
            update_count += 1

            if actor_loss_val is not None:
                avg_reward_this_batch_episodes = np.mean(rewards_for_current_batch_episodes) if rewards_for_current_batch_episodes else 0
                log_data["rewards"].append(avg_reward_this_batch_episodes)
                log_data["actor_losses"].append(actor_loss_val)
                log_data["critic_losses"].append(critic_loss_val)
                log_data["entropy"].append(entropy_val)

                if update_count % log_every_n_updates == 0:
                    print(f"\nUpdate: {update_count}, Global Steps: {global_step_count}, Avg Reward: {avg_reward_this_batch_episodes:.2f}")
                    print(f"Actor Loss: {actor_loss_val:.4f}, Critic Loss: {critic_loss_val:.4f}, Entropy: {entropy_val:.4f}")
            
            save_dir_path = Path(agent_params["save_dir"])
            if save_dir_path and update_count % save_model_every_n_updates == 0:
                agent.save_model(str(save_dir_path / f"ppo_update_{update_count}_steps_{global_step_count}"))

    progress_bar.close()
    if hasattr(env, 'close') and callable(env.close):
        env.close()
    print("Training finished.")
    return log_data, agent # Trả về log và agent đã huấn luyện


def run(env_params, agent, num_episodes=10, trained_agent_path=None):
    print("\nStarting PPO Agent Evaluation...")
    if trained_agent_path:
        # Logic để load agent từ trained_agent_path (cần PPOAgent có hàm load_model(path_to_checkpoint_file))
        # Ví dụ: agent.load_model(trained_agent_path) # Giả sử PPOAgent có hàm này
        # Hiện tại PPOAgent load trong __init__, nên có thể cần tạo agent mới với checkpoint_path
        print(f"Loading trained agent from: {trained_agent_path} (Not implemented in PPOAgent.evaluate_agent directly, loaded at init)")
        # For now, assume agent passed in is already loaded or new
        pass

    current_env_params = env_params.copy()
    max_eval_steps = current_env_params.pop("max_steps_per_episode", 200)
    env = Env(**current_env_params)
    num_hosts = env_params["NumHost"]
    original_host_state_shape = env.observation_space.shape
    expected_host_features_per_host = original_host_state_shape[0]

    total_rewards = []
    agent.actor_critic_net.eval() # Chuyển sang chế độ đánh giá

    for episode in range(num_episodes):
        original_host_state_np, info = env.reset()
        episode_reward = 0
        done_env = False
        truncated_env = False

        for step_num in range(max_eval_steps):
            selectable_vms_ids = env.getSelectableVms()
            if not selectable_vms_ids or done_env or truncated_env:
                break

            selected_vid = selectable_vms_ids[0]
            selected_vm_instance = env.getVmByID(selected_vid)
            vm_feats_np = get_vm_features(selected_vm_instance)
            
            current_ppo_specific_state_np = create_ppo_specific_state(
                original_host_state_np, vm_feats_np, num_hosts, expected_host_features_per_host
            )
            
            # Trong evaluate, thường chọn action một cách tham lam (deterministic) hoặc vẫn stochastic
            # PPOAgent.select_action hiện tại trả về action stochastic. 
            # Để có deterministic, PPOActorCriticNet.get_action_distribution cần có thêm option.
            # Hoặc, lấy mode/mean của distribution. For now, use stochastic from select_action.
            action_host_idx, _, _ = agent.select_action(current_ppo_specific_state_np) # Không cần log_prob, value
            
            decision = [(selected_vid, action_host_idx)]
            next_original_host_state_np, reward, done_env, truncated_env, info = env.step(decision)
            
            original_host_state_np = next_original_host_state_np
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}")

    avg_reward = np.mean(total_rewards)
    print(f"Average reward over {num_episodes} evaluation episodes: {avg_reward:.2f}")
    return avg_reward


# ----- MAIN EXECUTION -----
if __name__ == '__main__':
    # 1. Huấn luyện agent
    # print("===== TRAINING PHASE =====")
    # training_log_data, trained_ppo_agent = train(
    #     env_params=DEFAULT_ENV_PARAMS,
    #     agent_params=DEFAULT_PPO_AGENT_PARAMS,
    #     training_params=DEFAULT_TRAINING_PARAMS
    # )

    print("===== TEST TRAINING PHASE (CPU, small steps) =====")
    training_log_data, trained_ppo_agent = train(
        env_params=TEST_ENV_PARAMS,
        agent_params=TEST_PPO_AGENT_PARAMS,
        training_params=TEST_TRAINING_PARAMS
    )
    
    # 2. Vẽ đồ thị kết quả huấn luyện
    # plot_training_results(training_log_data, DEFAULT_PPO_AGENT_PARAMS["save_dir"])

    # 3. Đánh giá agent đã huấn luyện (ví dụ)
    # Bạn có thể muốn lưu agent tốt nhất và load nó ở đây,
    # hoặc dùng trực tiếp `trained_ppo_agent` vừa huấn luyện xong.
    print("\n===== TEST RUN PHASE =====")
    # Giả sử bạn muốn load model tốt nhất đã lưu (cần biết tên file cụ thể)
    # best_model_path = Path(DEFAULT_PPO_AGENT_PARAMS["save_dir"]) / "ppo_update_XXX_steps_YYY.chkpt" # Thay XXX, YYY
    # Hoặc tạo agent mới và load:
    # eval_agent_params = DEFAULT_PPO_AGENT_PARAMS.copy()
    # eval_agent_params["checkpoint_path_actor_critic"] = str(best_model_path) 
    # eval_agent = PPOAgent(state_dim=trained_ppo_agent.state_dim, 
    #                       action_dim=trained_ppo_agent.action_dim, 
    #                       **eval_agent_params)
    # evaluate_agent(DEFAULT_ENV_PARAMS, eval_agent, num_episodes=20)

    # Hoặc đánh giá trực tiếp agent vừa train xong:
    if trained_ppo_agent:
        # run(DEFAULT_ENV_PARAMS, trained_ppo_agent, num_episodes=20) 
        run(TEST_ENV_PARAMS, trained_ppo_agent, num_episodes=5)
    else:
        print("No trained agent available for evaluation.")