from envs.Env import Env
import numpy as np

class Scheduler():
    def __init__(self):
        self.env : Env = None

    def setEnvironment(self, env):
        self.env = env

    def AllVmSelection(self):
        return self.env.inactiveVmId

    def RandomVmSelection(self):
        selectableVmIds = self.env.getSelectableVms()
        if selectableVmIds == []:
            return []
        
        selectedCount = np.random.randint(0, len(selectableVmIds)) + 1
        seletedVmIds = []
        while (len(seletedVmIds)) < selectedCount:
            idChoice = np.randome.choice(selectableVmIds)
            if idChoice not in seletedVmIds:
                seletedVmIds.append(idChoice)
        return seletedVmIds
    
    def RamdomPlacement(self, vmIds : list[int]):
        decision = []
        for vmid in vmIds:
            decision.append((vmid, np.random.randint(0, len(self.env.hostlist))))
        return decision
    
    def FirstFitPlacement(self, vmIds : list[int]):
        decision = []
        for vmid in vmIds:
            for hostid in range(len(self.env.hostlist)):
                if self.env.checkPlacement(self.env.getVmByID(vmid), self.env.getHostByID(hostid)):
                    decision.append((vmid, hostid))
                    break
        return decision
    
    def RandomPlacement(self, vmIds : list[int]):
        decision = []
        for vmid in vmIds:
            decision.append((vmid, np.random.randint(0, len(self.env.hostlist))))
        return decision
    

    def LeastFullPlacement(self, vmIds : list[int]):
        decision = []
        hostIPSs = [(self.env.hostlist[i].getCoreAv(), i) for i in range(len(self.env.hostlist))]
        for vmid in vmIds:
            leastFullHost = min(hostIPSs)
            decision.append((vmid, leastFullHost[1]))
            if len(hostIPSs) > 1:
                hostIPSs.remove(leastFullHost)
        return decision

    def MaxFullPlacement(self, vmIds : list[int]):
        decision = []
        hostIPSs = [(self.env.hostlist[i].getCoreAv(), i) for i in range(len(self.env.hostlist))]
        for vmid in vmIds:
            maxFullHost = max(hostIPSs)
            decision.append((vmid, maxFullHost[1]))
            if len(hostIPSs) > 1:
                hostIPSs.remove(maxFullHost)
        return decision
    

    def getNumOfVms(self):
        return len(self.env.vmMatrix)