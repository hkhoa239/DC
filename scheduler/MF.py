

from .Scheduler import Scheduler
import numpy as np


class MFScheduler(Scheduler):
    def __init__(self, environment):
        super().__init__()
        self.setEnvironment(environment)
    
    def placement(self, vmIDs):
        return self.FirstFitPlacement(vmIDs)
    
    def run(self):
        vmIDs = self.AllVmSelection()
        return self.MaxFullPlacement(vmIDs)
    
    
    