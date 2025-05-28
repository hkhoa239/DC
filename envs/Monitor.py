import numpy as np

class Monitor():
    def __init__(self, environment):
        self.env = environment
        self.hostMatrix = []
        self.vmMatrix = []

        self.iW = 0.5
        self.pW = 0.5

        self.prev_ri = 0
        self.prev_rp = 0
        self.reward = 0

        self.imCore = 0
        self.imRam = 0  

    def createHostMatrix(self):
        matrix = []
        for host in self.env.hostlist:
            matrix.append([
                host.core,
                host.ram,
                host.coreUsed/host.core,
                host.ramUsed/host.ram
            ])
        self.hostMatrix = matrix
        return matrix

    def createVmMatrix(self):
        matrix = []
        for vmid in self.env.inactiveVmId:
            vm = self.env.vmlist[vmid]
            matrix.append([
                vm.core,
                vm.ram
            ])
        self.vmMatrix = matrix

    def initMatrix(self):
        self.createHostMatrix()
        self.createVmMatrix()

    def estimateImbalance(self):
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
        return imCore, imRam
        

    def estimateImbalanceMinMax(self):
        # minCore = float('inf')
        # minRam = float('inf')
        # maxCore = float('-inf')
        # maxRam = float('-inf')

        # for host 
        pass


    def estimateImRew(self, imCore, imRam, energy):
        self.imCore = imCore
        self.imRam = imRam
        r_i = (imCore + imRam)
        r_p = energy / self.env.totalpower
        self.prev_ri = r_i if self.prev_ri == 0 else self.prev_ri
        self.prev_rp = r_p if self.prev_rp == 0 else self.prev_rp
        r = self.iW * (r_i / self.prev_ri) + self.pW * (r_p / self.prev_rp) if r_i != 0 and r_p != 0 else 0
        self.prev_ri = r_i
        self.prev_rp = r_p
        self.reward = r
        return r

    def calcRew(self):
        imCore, imRam = self.estimateImbalance()
        energy = self.env.step_power
        self.reward = self.estimateImRew(imCore, imRam, energy)
        return self.reward
    
    def getState(self):
        state = []
        vmState = [0,0,0,0] if len(self.vmMatrix) == 0 else self.vmMatrix[0]
        for i in range(len(self.hostMatrix)):
            state.append([self.hostMatrix[i][2], self.hostMatrix[i][3], vmState[0]/self.hostMatrix[i][0], vmState[1]/self.hostMatrix[i][1]])
        state = np.array(state).T
        return state