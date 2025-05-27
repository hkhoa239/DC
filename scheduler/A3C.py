from .Scheduler import Scheduler
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import time
from envs.Env import Env

class ActorCritic(nn.Module):
    def __init__(self, input_dim, n_actions):
        super().__init__()

        input_size = np.prod(input_dim)

        self.fc = nn.Linear(input_size, 128)
        self.policy = nn.Linear(128, n_actions)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)

        x = F.relu(self.fc(x))
        return self.policy(x), self.value(x)

# This one is not currently used, but kept for reference
def worker(global_model, optimizer, env, counter, max_step):
    hostMatrix = env.getHostMatrix()
    vmMatrix = env.getVmMatrix()

    def getState(hostMatrix, vmState):
        state = []
        for i in range(len(hostMatrix)):
            state.append([hostMatrix[i][2], hostMatrix[i][3], vmState[0]/hostMatrix[i][0], vmState[1]/hostMatrix[i][1]])
        state = np.array(state).T
        return state

    def updateWithAction(state, env, vmid, hostid, hostMatrix):
        vm = env.getVmByID(vmid)
        hostMatrix[hostid][2] += vm.core / hostMatrix[hostid][0]
        hostMatrix[hostid][3] += vm.ram / hostMatrix[hostid][1]

        state[2][hostid] = hostMatrix[hostid][2]
        state[3][hostid] = hostMatrix[hostid][3]
        rew = self.estimateReward()
        return state, rew, False


    local_model = ActorCritic(env.observation_space.shape, env.action_space.n)
    local_model.load_state_dict(global_model.state_dict())


    for episode in range(NUM_EP):
        hostMatrix = env.getHostMatrix()
        vmMatrix = env.getVmMatrix()
        decision = []        
        rew = 0
        for i, vmState in enumerate(vmMatrix):
            state = getState(hostMatrix, vmState)
            vmid = env.inactiveVmId[i]
            vm = env.getVmByID(vmid)

            state = torch.tensor(state, device='cpu').unsqueeze(0)
            actions_values, _ = local_model(state)
            action_idx = torch.argmax(actions_values, axis=1).item()

            next_state, reward, done = self.updateWithAction(state, vmid, action)

"""
        log_probs, values, rewards = [], [], []
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).squeeze(0)
            print(state_tensor.shape)
            time.sleep(5)
            logits, value = local_model(state_tensor)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            item = action.item()
            next_state, reward, done, _ = env.step(item)

            log_probs.append(dist.log_prob(action))
            values.append(value)
            rewards.append(reward)

            state = next_state
            with counter.get_lock():
                counter.value += 1
                if counter.value >= max_step:
                    return

    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + 0.99 * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32)
    values = torch.cat(values)
    log_probs = torch.stack(log_probs)
    advantage = returns - values.squeeze()

    policy_loss = -(log_probs * advantage.detatch()).mean()
    value_loss = advantage.pow(2).mean()
    loss = policy_loss + 0.5 * value_loss

    optimizer.zero_grad()
    loss.backword()
    for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
        global_param._grad = local_param.grad
    optimizer.step()
    local_model.load_state_dict(global_model.state_dict())
"""

class A3CScheduler(Scheduler):
    def __init__(self, environment):
        super().__init__()

        self.iW = 0.5
        self.pW = 0.5
        self.prev_ri = 0
        self.prev_rp = 0


        self.num_episodes = 1000
        self.exploration_rate = 0.02
        self.max_steps = 10000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setEnvironment(environment)

    def getStateOfEnv(self):
        return self.env.getState()

    def getObservation(self):
        self.hostMatrix = self.env.getHostMatrix()
        self.vmMatrix = self.env.getVmMatrix()

    def getState(self, vmState):
        state = []
        for i in range(len(self.hostMatrix)):
            state.append([self.hostMatrix[i][2], self.hostMatrix[i][3], vmState[0]/self.hostMatrix[i][0], vmState[1]/self.hostMatrix[i][1]])
        state = np.array(state).T
        return state

    @staticmethod
    def estimateReward(hostMatrix, env, iW=0.5, pW=0.5, prev_ri=0, prev_rp=0):
        totalCore = 0
        for host in hostMatrix:
            totalCore += host[2]
        avgCore = totalCore / len(env.hostlist)
        imCore = 0
        for host in hostMatrix:
            imCore += (host[2]/avgCore-1)**2 if avgCore != 0 else 0
        imCore = np.sqrt(imCore / len(env.hostlist))

        totalRam = 0
        for host in hostMatrix:
            totalRam += host[3]
        avgRam = totalRam / len(env.hostlist)
        imRam = 0
        for host in hostMatrix:
            imRam += (host[3]/avgRam - 1)**2 if avgRam != 0 else 0
        imRam = np.sqrt(imRam / len(env.hostlist))

        powerScore = 0
        for host in hostMatrix:
            powerScore += host[0] * host[2]

        r_i = (imCore + imRam)
        r_p = powerScore / env.totalpower
        prev_ri = r_i if prev_ri == 0 else prev_ri
        prev_rp = r_p if prev_rp == 0 else prev_rp
        r = -iW * (r_i / prev_ri) + -pW * (r_p / prev_rp) if r_i != 0 and r_p != 0 else 0
        prev_ri = r_i 
        prev_rp = r_p 
        return r

    @staticmethod
    def updateWithAction(state, vmid, hostid, hostMatrix, env):
        vm = env.getVmByID(vmid)
        hostMatrix[hostid][2] += vm.core / hostMatrix[hostid][0]
        hostMatrix[hostid][3] += vm.ram / hostMatrix[hostid][1]

        state[2][hostid] = hostMatrix[hostid][2]
        state[3][hostid] = hostMatrix[hostid][3]
        rew = A3CScheduler.estimateReward(hostMatrix, env)
        return state, rew, False

    @staticmethod
    def worker(global_model, optimizer, counter, max_steps, num_episodes, exploration_rate):
        env = Env()
        env.reset()
        local_model = ActorCritic(env.observation_space.shape, env.action_space.n)
        local_model.load_state_dict(global_model.state_dict())
        for e in range(num_episodes):
            hostMatrix = env.getHostMatrix()
            vmMatrix = env.getVmMatrix()
            rew = 0
            decision = []
            log_probs, values, rewards = [], [], []
            for i, vmState in enumerate(vmMatrix):

                state = []
                for j in range(len(hostMatrix)):
                    state.append([hostMatrix[j][2], hostMatrix[j][3], vmState[0]/hostMatrix[j][0], vmState[1]/hostMatrix[j][1]])
                state = np.array(state).T
                state_tensor = torch.tensor(state, dtype=torch.float32, device="cpu").unsqueeze(0)
                logits, value = local_model(state_tensor)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                if np.random.rand() < exploration_rate:
                    action = torch.tensor([np.random.randint(env.action_space.n)])
                else:
                    action = dist.sample()
                vmid = env.inactiveVmId[i]
                next_state, reward, done = A3CScheduler.updateWithAction(state, vmid, action, hostMatrix, env)

                log_probs.append(dist.log_prob(action))
                values.append(value)
                rewards.append(reward)
                rew += reward
                decision.append((vmid, action))
            
            print(f"{counter.value},", "Decision: ", decision)

            R = 0
            returns = []
            for r in reversed(rewards):
                R = r + 0.99 * R # gamma = 0.99
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32)
            values = torch.cat(values)
            log_probs = torch.stack(log_probs)
            advantage = returns - values.squeeze()

            # A3C Loss
            policy_loss = -(log_probs * advantage.detach()).mean()
            value_loss = advantage.pow(2).mean()
            loss = policy_loss + 0.5 * value_loss

            rew = rew/len(vmMatrix) if len(vmMatrix) > 0 else 0
            optimizer.zero_grad()
            loss.backward()
            for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
                global_param._grad = local_param.grad            
            optimizer.step()
            local_model.load_state_dict(global_model.state_dict())
            with counter.get_lock():
                env.step(decision)
                counter.value += 1
                if counter.value >= max_steps:
                    return



    def run(self):
        self.global_model = ActorCritic(input_dim=self.env.observation_space.shape, n_actions=self.env.action_space.n)
        self.global_model.share_memory()
        self.optimizer = optim.Adam(self.global_model.parameters(), lr=1e-3)
        self.counter = mp.Value('i',0)
        self.max_steps = 10000

        processes = []
        num_workers = 16
        for _ in range(num_workers):
            p = mp.Process(target=A3CScheduler.worker, args=(
                self.global_model, self.optimizer, self.counter, self.max_steps,
                self.num_episodes, self.exploration_rate
            ))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        vmIDs = self.AllVmSelection()
        return self.FirstFitPlacement(vmIDs)