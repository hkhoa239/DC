from .Scheduler import Scheduler
from .rl.agent.DuelingDDQLAgent import DuelingDDQLAgent
from .rl.Logger import Logger
from pathlib import Path
import datetime
import numpy as np

class DuelingDDQLScheduler(Scheduler):
    def __init__(self, environment, checkpoint_path=""):
        super().__init__()
        self.episodes = 50_000

        self.iW = 0.5
        self.pW = 0.5
        self.prev_ri = 0
        self.prev_rp = 0

        self.setEnvironment(environment)
        self.setAgent(environment, checkpoint_path)

    def setAgent(self, env, checkpoint_path=""):
        self.save_dir = Path(f"train_ddql") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.agent = DuelingDDQLAgent(state_dim=env.observation_space.shape, action_dim=env.action_space.n, save_dir=self.save_dir, checkpoint=checkpoint_path)

    def getObservation(self):
        self.hostMatrix = self.env.getHostMatrix()
        self.vmMatrix = self.env.getVmMatrix()

    def getState(self, vmState):
        state = []
        for i in range(len(self.hostMatrix)):
            state.append([self.hostMatrix[i][2], self.hostMatrix[i][3], vmState[0]/self.hostMatrix[i][0], vmState[1]/self.hostMatrix[i][1]])
        state = np.array(state).T
        return state

    def estimateReward(self):
        totalCore = 0
        for host in self.hostMatrix:
            totalCore += host[2]
        avgCore = totalCore / len(self.env.hostlist)
        imCore = 0
        for host in self.hostMatrix:
            imCore += (host[2]/avgCore-1)**2 if avgCore != 0 else 0
        imCore = np.sqrt(imCore / len(self.env.hostlist))

        totalRam = 0
        for host in self.hostMatrix:
            totalRam += host[3]
        avgRam = totalRam / len(self.env.hostlist)
        imRam = 0
        for host in self.hostMatrix:
            imRam += (host[3]/avgRam - 1)**2 if avgRam != 0 else 0
        imRam = np.sqrt(imRam / len(self.env.hostlist))

        # powerScore = 0
        # for host in self.hostMatrix:
        #     powerScore += host[0] * host[2]

        r_i = (imCore + imRam)
        # r_p = powerScore / self.env.totalpower
        # self.prev_ri = r_i if self.prev_ri == 0 else self.prev_ri
        # self.prev_rp = r_p if self.prev_rp == 0 else self.prev_rp
        # r = -self.iW * (r_i / self.prev_ri) + -self.pW * (r_p / self.prev_rp) if r_i != 0 and r_p != 0 else 0
        # self.prev_ri = r_i 
        # self.prev_rp = r_p 
        return -r_i

    def updateWithAction(self, state, vmid, hostid):
        vm = self.env.getVmByID(vmid)
        self.hostMatrix[hostid][2] += vm.core / self.hostMatrix[hostid][0]
        self.hostMatrix[hostid][3] += vm.ram / self.hostMatrix[hostid][1]

        state[2][hostid] = vm.core / self.hostMatrix[hostid][0]
        state[3][hostid] = vm.ram / self.hostMatrix[hostid][1]
        rew = self.estimateReward()
        return state, rew, False


    def train(self):
        self.save_dir.mkdir(parents=True)

        logger = Logger(save_dir=self.save_dir)
        self.env.reset()
        self.getObservation()
        for e in range(self.episodes):
            self.getObservation()
            decision = []
            rew = 0
            for i, vmState in enumerate(self.vmMatrix):
                state = self.getState(vmState)
                vmid = self.env.inactiveVmId[i]
                vm = self.env.getVmByID(vmid)
                
                action = self.agent.act(state)
                next_state, reward, done = self.updateWithAction(state, vmid, action)
                self.agent.cache(state, next_state, action, reward, done)
                q, loss = self.agent.learn()
                rew += reward
                decision.append((vmid, action))

            rew = rew/len(self.vmMatrix) if len(self.vmMatrix) > 0 else 0
            logger.log_step(rew, loss, q, e)
            self.env.step(decision)

            if e % 100 == 0:
                logger.log_episode()
                logger.record(episode=e, epsilon=self.agent.exploration_rate, step=self.agent.curr_step)
            if e % 1000 == 0:
                self.env.reset()
        self.env.initialized_flag = False

    def run(self):
        decision = []
        self.getObservation()
        for i, vmState in enumerate(self.vmMatrix):
            state = self.getState(vmState)
            action = self.agent.predict(state)
            vmid = self.env.inactiveVmId[i]
            decision.append((vmid, action))
            self.updateWithAction(state, vmid, action)
        return decision