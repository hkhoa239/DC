

class Workload():
    def __init__(self):
        self.creation_id = 0
        self.createdVms = []
        self.deployedVms = []

    def getUndeployedVms(self):
        undeployed = []
        for i, deployed in enumerate(self.deployedVms):
            if not deployed:
                undeployed.append(self.createdVms[i])

        return undeployed
    
    def reset(self):
        self.creation_id = 0
        self.createdVms = []
        self.deployedVms = []

    def updateDeployedVms(self, creationIds):
        for vid in creationIds:
            self.deployedVms[vid] = True
            