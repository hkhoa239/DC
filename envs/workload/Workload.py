import numpy as np
# from FibHeap import FibonacciHeap

class Workload():
    def __init__(self):
        self.creation_id = 0
        # self.heap = FibonacciHeap()
        # self.self.containerlist = []
        self.createdContainers = []
        self.deployedContainers = []

    # def getUndeployedContainers(self):
    #     undeployed = []

    #     for i,deployed in enumerate(self.deployedContainers):
    #         if not deployed:
    #             undeployed.append(self.createdContainers[i])

    #     return undeployed
    
    # def updateDeployedContainers(self, creationIDs):
    #     for cid in creationIDs:
    #         assert not self.deployedContainers[cid]
    #         self.deployedContainers[cid] = True