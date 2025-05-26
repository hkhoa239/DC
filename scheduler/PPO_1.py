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

# ----- HYPERPARAMETERS -----
DEFAULT_ENV_PARAMS = {
    "TotalPower": 2000,
    "RouterBw": 10000,
    "NumHost": 10, # Default NumHost nếu không được cung cấp cụ thể
    "VmLimit": 50,
    "IntervalTime": 1
}

MAX_STEPS_PER_PPO_EPISODE = 200 # Mặc định cho huấn luyện đầy đủ

DEFAULT_PPO_AGENT_PARAMS = {
    "lr_actor_critic": 3e-4,
    "gamma": 0.99,
    "ppo_epochs": 10,
    "clip_epsilon": 0.2,
    "gae_lambda": 0.95,
    "entropy_coef": 0.01,
    "vf_coef": 0.5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir": "ppo_checkpoints_synced_state_v2", # Thay đổi thư mục lưu
    "checkpoint_path_actor_critic": None
}

DEFAULT_TRAINING_PARAMS = {
    "total_training_steps": 1_000_000,
    "steps_per_batch_collection": 2048,
    "save_model_every_n_updates": 50,
    "log_every_n_updates": 10
}


# def plot_training_results(log_data, save_dir):
#     episode_rewards_log = log_data.get("rewards", [])
#     actor_losses_log = log_data.get("actor_losses", [])
#     critic_losses_log = log_data.get("critic_losses", [])
#     entropy_log = log_data.get("entropy", [])

#     if not any([episode_rewards_log, actor_losses_log, critic_losses_log, entropy_log]):
#         print("No data to plot for PPO.")
#         return

#     save_plot_dir = Path(save_dir or ".")
#     save_plot_dir.mkdir(parents=True, exist_ok=True)

#     plt.figure(figsize=(18, 6))
#     if episode_rewards_log:
#         plt.subplot(1, 3, 1); plt.plot(episode_rewards_log); plt.title("PPO: Avg Reward per Batch"); plt.xlabel("Update");
    
#     if actor_losses_log or critic_losses_log:
#         plt.subplot(1, 3, 2);
#         if actor_losses_log: plt.plot(actor_losses_log, label="Actor Loss")
#         if critic_losses_log: plt.plot(critic_losses_log, label="Critic Loss")
#         plt.title("PPO: Losses"); plt.xlabel("Update"); plt.legend()
    
#     if entropy_log:
#         plt.subplot(1, 3, 3); plt.plot(entropy_log); plt.title("PPO: Policy Entropy"); plt.xlabel("Update");
    
#     plt.tight_layout()
#     file_name = "ppo_training_curves_state_dim_fix.png"
#     plt.savefig(save_plot_dir / file_name)
#     print(f"PPO training curves saved to {save_plot_dir / file_name}")

# ----- TRAIN FUNCTION -----
def train(env_params, agent_params, training_params, max_steps_ppo_ep):
    print("Starting PPO Training (State dim derived from actual matrices)...")
    print(f"Environment Params: {env_params}")
    print(f"Agent Params: {agent_params}")
    print(f"Training Params: {training_params}")
    print(f"Max steps per PPO episode: {max_steps_ppo_ep}")

    env = Env(**env_params)

    # --- Tính toán ppo_state_dim một cách linh hoạt ---
    _ = env.reset() 
    sample_host_matrix = env.getHostMatrix()
    sample_vm_matrix = env.getVmMatrix()

    actual_num_hosts_in_env = 0
    num_host_features = 0
    num_vm_features = 0

    if sample_host_matrix and isinstance(sample_host_matrix, list) and len(sample_host_matrix) > 0 and \
       isinstance(sample_host_matrix[0], list) and len(sample_host_matrix[0]) > 0:
        actual_num_hosts_in_env = len(sample_host_matrix)
        num_host_features = len(sample_host_matrix[0])
    else:
        # Fallback nếu host_matrix không hợp lệ hoặc rỗng lúc đầu
        actual_num_hosts_in_env = env_params.get("NumHost", DEFAULT_ENV_PARAMS["NumHost"]) # Lấy từ params
        num_host_features = 4 # Dựa trên Monitor.createHostMatrix
        print(f"Warning: sample_host_matrix was empty or malformed. Using NumHost={actual_num_hosts_in_env}, host_features={num_host_features}.")

    if sample_vm_matrix and isinstance(sample_vm_matrix, list) and len(sample_vm_matrix) > 0 and \
       isinstance(sample_vm_matrix[0], list) and len(sample_vm_matrix[0]) > 0:
        num_vm_features = len(sample_vm_matrix[0])
    else:
        # Fallback nếu vm_matrix không hợp lệ hoặc rỗng lúc đầu
        num_vm_features = 2 # Dựa trên Monitor.createVmMatrix
        print(f"Warning: sample_vm_matrix was empty or malformed. Using vm_features={num_vm_features}.")
        
    ppo_state_dim = (actual_num_hosts_in_env * num_host_features) + num_vm_features
    # --- Kết thúc tính toán ppo_state_dim ---

    action_dim = env.action_space.n 

    print(f"PPO specific state_dim (derived from actual matrices): {ppo_state_dim} "
          f"(Hosts: {actual_num_hosts_in_env} x Feat: {num_host_features} + VM Feat: {num_vm_features})")
    
    # Khởi tạo agent với state_dim đã được tính toán chính xác
    agent = PPOAgent(state_dim=ppo_state_dim, action_dim=action_dim, **agent_params)

    log_data = {"rewards": [], "actor_losses": [], "critic_losses": [], "entropy": []}
    global_step_count = 0
    update_count = 0
    
    training_params_dict = training_params
    total_training_steps = training_params_dict["total_training_steps"]
    steps_per_batch_collection = training_params_dict["steps_per_batch_collection"]
    save_model_every_n_updates = training_params_dict["save_model_every_n_updates"]
    log_every_n_updates = training_params_dict["log_every_n_updates"]

    progress_bar = tqdm(total=total_training_steps, desc=f"PPO Training (StateDim:{ppo_state_dim})")
    
    while global_step_count < total_training_steps:
        num_steps_collected_this_batch = 0
        rewards_for_current_batch_episodes = []

        while num_steps_collected_this_batch < steps_per_batch_collection:
            _, _ = env.reset() 
            host_matrix_full = env.getHostMatrix()
            vm_matrix_full = env.getVmMatrix()
            inactive_vm_ids = env.getSelectableVms()

            current_ppo_episode_reward_sum = 0
            num_vms_to_place_this_env_step = len(inactive_vm_ids)

            for i in range(min(num_vms_to_place_this_env_step, max_steps_ppo_ep - (global_step_count % max_steps_ppo_ep if max_steps_ppo_ep > 0 else 0) )):
                if global_step_count >= total_training_steps or \
                   num_steps_collected_this_batch >= steps_per_batch_collection:
                    break
                
                if i >= len(vm_matrix_full) or not host_matrix_full: 
                    break

                vm_state_for_ppo = vm_matrix_full[i] 
                selected_vid = inactive_vm_ids[i]

                temp_state_parts = []
                for host_row in host_matrix_full:
                    temp_state_parts.extend(host_row) 
                temp_state_parts.extend(vm_state_for_ppo) 

                try:
                    current_ppo_specific_state_np = np.array(temp_state_parts, dtype=np.float32)
                except ValueError as e:
                    print(f"\nError converting state parts to np.array for VM ID {selected_vid}: {e}")
                    print(f"Host matrix part ({len(host_matrix_full)} hosts, {len(host_matrix_full[0] if host_matrix_full else 0)} features): {host_matrix_full}")
                    print(f"VM state part ({len(vm_state_for_ppo)} features): {vm_state_for_ppo}")
                    print(f"Combined temp_state_parts (len {len(temp_state_parts)}): {temp_state_parts}")
                    # Có thể bỏ qua bước này hoặc xử lý lỗi khác
                    continue # Bỏ qua VM này nếu state không hợp lệ

                if current_ppo_specific_state_np.shape[0] != ppo_state_dim:
                    print(f"\nRuntime State Dimension Mismatch!")
                    print(f"Expected ppo_state_dim: {ppo_state_dim}")
                    print(f"Actual current_ppo_specific_state_np dimension: {current_ppo_specific_state_np.shape[0]}")
                    print(f"Host matrix ({len(host_matrix_full)} hosts, {len(host_matrix_full[0] if host_matrix_full else 0)} features)")
                    print(f"VM state ({len(vm_state_for_ppo)} features)")
                    # Nên dừng hoặc xử lý lỗi này
                    raise ValueError("Runtime state dimension mismatch with PPO agent initialization.")

                action_host_idx, log_prob_action_tensor, value_estimate_tensor = \
                    agent.select_action(current_ppo_specific_state_np)
                
                decision_for_one_vm = [(selected_vid, action_host_idx)]
                state_for_memory_tensor = torch.tensor(current_ppo_specific_state_np, dtype=torch.float32, device=agent.device)

                _, reward, done_env, truncated_env, _ = env.step(decision_for_one_vm)
                
                next_host_matrix_full = env.getHostMatrix()
                next_vm_matrix_full = env.getVmMatrix()
                next_inactive_vm_ids = env.getSelectableVms()

                is_ppo_transition_terminal = done_env or truncated_env
                current_ppo_episode_step_count = (global_step_count % max_steps_ppo_ep if max_steps_ppo_ep > 0 else 0) + 1
                if max_steps_ppo_ep > 0 and current_ppo_episode_step_count == max_steps_ppo_ep :
                    is_ppo_transition_terminal = True
                
                agent._store_transition_tensor(
                    state_for_memory_tensor,
                    torch.tensor([action_host_idx], dtype=torch.long, device=agent.device),
                    log_prob_action_tensor, value_estimate_tensor,
                    torch.tensor([reward], dtype=torch.float32, device=agent.device),
                    torch.tensor([is_ppo_transition_terminal], dtype=torch.bool, device=agent.device)
                )
                
                host_matrix_full = next_host_matrix_full 
                vm_matrix_full = next_vm_matrix_full
                inactive_vm_ids = next_inactive_vm_ids

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
                if inactive_vm_ids and vm_matrix_full: # Đảm bảo có VM tiếp theo
                    next_vm_state_for_gae = vm_matrix_full[0] 
                    temp_next_state_parts = []
                    if host_matrix_full: # Đảm bảo host_matrix_full không rỗng
                        for host_row in host_matrix_full:
                            temp_next_state_parts.extend(host_row)
                    temp_next_state_parts.extend(next_vm_state_for_gae)
                    try:
                        next_ppo_state_for_gae_value_np = np.array(temp_next_state_parts, dtype=np.float32)
                        if next_ppo_state_for_gae_value_np.shape[0] != ppo_state_dim:
                            print(f"\nWarning: GAE next_state dim mismatch. Expected {ppo_state_dim}, got {next_ppo_state_for_gae_value_np.shape[0]}. Setting to None.")
                            next_ppo_state_for_gae_value_np = None # Không dùng state này nếu dim sai
                    except ValueError:
                         print(f"\nWarning: Error converting GAE next_state. Setting to None.")
                         next_ppo_state_for_gae_value_np = None


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
                    print(f"PPO Actor Loss: {actor_loss_val:.4f}, Critic Loss: {critic_loss_val:.4f}, Entropy: {entropy_val:.4f}")
            
            save_dir_path = Path(agent_params["save_dir"])
            if save_dir_path and update_count % save_model_every_n_updates == 0:
                agent.save_model(str(save_dir_path / f"ppo_update_{update_count}_steps_{global_step_count}"))

    progress_bar.close()
    if hasattr(env, 'close') and callable(env.close):
        env.close()
    print(f"PPO Training (StateDim:{ppo_state_dim}) finished.")
    return log_data, agent

# ----- EVALUATE FUNCTION -----
def evaluate_agent(env_params, agent, ppo_state_dim_eval, num_episodes=10, max_steps_ppo_ep=200):
    print(f"\nStarting PPO Agent Evaluation (StateDim:{ppo_state_dim_eval})...")
    
    env = Env(**env_params)
    total_rewards = []
    if hasattr(agent, 'actor_critic_net'): agent.actor_critic_net.eval()

    for episode_idx in range(num_episodes):
        _, _ = env.reset()
        host_matrix_full = env.getHostMatrix()
        vm_matrix_full = env.getVmMatrix()
        inactive_vm_ids = env.getSelectableVms()
        
        episode_reward = 0
        done_env, truncated_env = False, False
        num_vms_to_place_this_env_step = len(inactive_vm_ids)

        for i in range(min(num_vms_to_place_this_env_step, max_steps_ppo_ep)):
            if done_env or truncated_env or i >= len(vm_matrix_full) or not host_matrix_full:
                break

            vm_state_for_ppo = vm_matrix_full[i]
            selected_vid = inactive_vm_ids[i]

            temp_state_parts = []
            for host_row in host_matrix_full:
                temp_state_parts.extend(host_row)
            temp_state_parts.extend(vm_state_for_ppo)
            
            try:
                current_ppo_specific_state_np = np.array(temp_state_parts, dtype=np.float32)
            except ValueError:
                print(f"Eval Error: Skipping VM {selected_vid} due to state conversion error.")
                continue
            
            if current_ppo_specific_state_np.shape[0] != ppo_state_dim_eval:
                 print(f"Eval Error: State dim mismatch for VM {selected_vid}. Expected {ppo_state_dim_eval}, got {current_ppo_specific_state_np.shape[0]}. Skipping.")
                 continue
            
            action_host_idx, _, _ = agent.select_action(current_ppo_specific_state_np)
            
            decision_for_one_vm = [(selected_vid, action_host_idx)]
            _, reward, done_env, truncated_env, _ = env.step(decision_for_one_vm)
            
            host_matrix_full = env.getHostMatrix()
            vm_matrix_full = env.getVmMatrix()
            inactive_vm_ids = env.getSelectableVms()
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        print(f"PPO Eval Episode {episode_idx + 1}/{num_episodes}, Reward: {episode_reward:.2f}")

    avg_reward = np.mean(total_rewards) if total_rewards else 0
    print(f"PPO Average reward over {num_episodes} evaluation episodes: {avg_reward:.2f}")
    return avg_reward

# ----- MAIN EXECUTION -----
if __name__ == '__main__':
    TEST_ENV_PARAMS = {
        "TotalPower": 2000, "RouterBw": 10000, "NumHost": 5,
        "VmLimit": 10, "IntervalTime": 1
    }
    TEST_MAX_STEPS_PPO_EP = 50 

    TEST_PPO_AGENT_PARAMS = DEFAULT_PPO_AGENT_PARAMS.copy()
    TEST_PPO_AGENT_PARAMS["device"] = "cpu"
    TEST_PPO_AGENT_PARAMS["save_dir"] = "ppo_checkpoints_test_synced_v2"

    TEST_TRAINING_PARAMS = {
        "total_training_steps": 1000, "steps_per_batch_collection": 128,
        "save_model_every_n_updates": 10, "log_every_n_updates": 5
    }

    print("===== PPO TEST TRAINING PHASE (State dim derived, CPU, small steps) =====")
    # Tính ppo_state_dim một lần ở đây để truyền vào evaluate nếu cần
    temp_env_for_dim_check = Env(**TEST_ENV_PARAMS)
    _ = temp_env_for_dim_check.reset()
    shm = temp_env_for_dim_check.getHostMatrix()
    svm = temp_env_for_dim_check.getVmMatrix()
    
    ahn = 0; nhf = 0; nvf = 0
    if shm and isinstance(shm, list) and shm[0] and isinstance(shm[0], list):
        ahn = len(shm)
        nhf = len(shm[0])
    else: ahn = TEST_ENV_PARAMS.get("NumHost"); nhf = 4; print("Warning: Using default host structure for dim check.")
    if svm and isinstance(svm, list) and svm[0] and isinstance(svm[0], list):
        nvf = len(svm[0])
    else: nvf = 2; print("Warning: Using default VM structure for dim check.")
    
    actual_ppo_state_dim_for_run = (ahn * nhf) + nvf
    del temp_env_for_dim_check # Xóa env tạm

    training_log_data, trained_ppo_agent = train(
        env_params=TEST_ENV_PARAMS,
        agent_params=TEST_PPO_AGENT_PARAMS,
        training_params=TEST_TRAINING_PARAMS,
        max_steps_ppo_ep=TEST_MAX_STEPS_PPO_EP
    )
    
    # plot_training_results(training_log_data, TEST_PPO_AGENT_PARAMS["save_dir"])

    if trained_ppo_agent:
        print("\n===== PPO TEST EVALUATION PHASE (State dim derived) =====")
        evaluate_agent(
            TEST_ENV_PARAMS,
            trained_ppo_agent,
            ppo_state_dim_eval=actual_ppo_state_dim_for_run, # Truyền state_dim đã tính
            num_episodes=5,
            max_steps_ppo_ep=TEST_MAX_STEPS_PPO_EP
        )
    else:
        print("No trained PPO agent available for evaluation from the test run.")
