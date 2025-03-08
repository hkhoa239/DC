from .Workload import Workload
from random import gauss, randint
import pandas as pd
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from container.IPSModels.IPSMBitbrain import IPSMBitbrain
from container.RAMModels.RMBitbrain import RMBitbrain
from container.DiskModels.DMBitbrain import DMBitbrain

# Intel Pentium III gives 2054 MIPS at 600 MHz
ips_multiplier = 2054.0 / (2 * 600)

class BWGD(Workload):
    def __init__(self, meanNumContainers, sigmaNumContainers):
        super().__init__()
        self.mean = meanNumContainers
        self.sigma = sigmaNumContainers
        self.meanSLA, self.sigmaSLA = 20, 3
        


        self.dataset_path = "/Users/huynhledangkhoa/Documents/NCKH/data/dc/envs/dataset/gwa-bitbrains/fastStorage"
        # self.dataset_path = os.path.abspath(os.path.join(parent_dir, "../dataset/gwa-bitbrains/fastStorage"))
        # print("Parent dir: ", parent_dir)
        # ("dataset: ", self.dataset_path)

        self.disk_sizes = [100, 200, 300, 400, 500]


    def generateNewContainers(self, interval):
        workloadlist = []
        for i in range(max(1,int(gauss(self.mean, self.sigma)))):
            CreationID = self.creation_id
            index = randint(1,500)
            
            df = pd.read_csv(self.dataset_path+"/"+str(index)+'.csv', sep=';\t', engine='python')
            
            sla = gauss(self.meanSLA, self.sigmaSLA)

            IPSModel = IPSMBitbrain(
                ips_list=(ips_multiplier*df['CPU usage [MHZ]']).to_list(), 
                max_ips=(ips_multiplier*df['CPU capacity provisioned [MHZ]'].to_list()[0]), 
                duration=int(1.2*sla), 
                SLA=sla
            )

            RAMModel = RMBitbrain(
                size_list=(df['Memory usage [KB]']/1000).to_list(), 
                read_list=(df['Network received throughput [KB/s]']/1000).to_list(),
                write_list=(df['Network transmitted throughput [KB/s]']/1000).to_list()
            )
            
            disk_size = self.disk_sizes[index % len(self.disk_sizes)]

            DiskModel = DMBitbrain(
                constant_size=disk_size,
                read_list=(df['Disk read throughput [KB/s]']/1000).to_list(),
                write_list=(df['Disk write throughput [KB/s]']/1000).to_list()
            )

            workloadlist.append((CreationID, interval, IPSModel, RAMModel, DiskModel))
            
            self.creation_id += 1

        
        self.createdContainers += workloadlist

        # self.deployedContainers += [False] * len(workloadlist)
            
        return workloadlist

