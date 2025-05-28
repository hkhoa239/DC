from .Workload import Workload
import pandas as pd
from random import gauss, randint
# Intel Pentium III gives 2054 MIPS at 600 MHz
# Source: https://archive.vn/20130205075133/http://www.tomshardware.com/charts/cpu-charts-2004/Sandra-CPU-Dhrystone,449.html
ips_multiplier = 2054.0 / (2 * 600)

class Bitbrain(Workload):
    def __init__(self, meanNumVms, sigmaNumVms, meanEndTime=20, sigmaEndTime=200, meanSLA=20, sigmaSLA=3):
        super().__init__()
        self.mean = meanNumVms
        self.sigma = sigmaNumVms
        self.meanEndTime = meanEndTime
        self.sigmaEndTime = sigmaEndTime
        
        self.meanSLA, self.sigmaSLA = meanSLA, sigmaSLA
        # self.path = '/bitbrain'
        # self.indices = []
        # for i in range(1, 500):
        #     df = pd.read_csv(self.path + str(i) + '.csv', sep=';\t')
        #     if (ips_multiplier*df['CPU usage [MHZ]']).to_list()[10]<3000 and (ips_multiplier*df['CPU usage [MHZ]']).to_list()[10]>500:
        #         self.indices.append(i)


        self.vm_conf = {
			'cpu' : ['a1', 'a1', 'a1', 'a1', 'a1', 'a1'], # CPU Type
			'core' : [2, 4, 8, 16, 4, 16, 8], # GB
			# 'RAMRead' : [3000, 2000, 3000],
			# 'RAMWrite' : [3000, 2000, 3000],
			'ram' : [4096, 8192, 16384, 32768, 16384, 32768, 32768],
			# 'DiskRead' : [2000, 2000, 3000],
			# 'DiskWrite' : [2000, 2000, 3000],
			'bwUp' : [1000, 1000, 1000, 1000, 1000, 1000, 1000],
			'bwDown': [1000, 1000, 1000, 1000, 1000, 1000, 1000],
 		}


    def generateNewVms(self, interval):
        vmlist = []
        # for i in range(int(self.mean)):
        for i in range(max(1, int(gauss(self.mean, self.sigma)))):
            creationId = self.creation_id
            endtime = self.meanEndTime + interval
            # endtime = max(1, int(gauss(self.meanEndTime, self.sigmaEndTime))) + interval
            sla = self.meanSLA, self.sigmaSLA
            # sla = max(1, int(gauss(self.meanSLA, self.sigmaSLA)))
            typeID = creationId % (len(self.vm_conf))
            # typeID = randint(0,len(self.vm_conf)-1)
            Cpu = self.vm_conf['cpu'][typeID]
            Core = self.vm_conf['core'][typeID]
            Ram = self.vm_conf['ram'][typeID]
            # Disk_ = self.vm_conf['DiskSize'][typeID]
            Bw_Up = self.vm_conf['bwUp'][typeID]
            Bw_Down = self.vm_conf['bwDown'][typeID]

            vmlist.append((creationId, interval, sla, endtime, Cpu, Core, Ram, Bw_Down, Bw_Down))
            self.creation_id += 1
        self.createdVms += vmlist
        self.deployedVms += [False] * len(vmlist)
        return self.getUndeployedVms()


    def destroyVms(self, interval):
        cids = []
        after1 = []
        after2 = []
        self.vmCopy = self.createdVms.copy()
        for i, (creationId, interv, sla, endtime, Cpu, Core, Ram, Bw_Down, Bw_Down) in enumerate(self.vmCopy):
            if endtime <= interval:
                cids.append(creationId)
        return cids


    def generateScenario(self, ep=1000):
        
        # for i in range(int(self.mean)):
        for interval in range(ep):
            vmlist = []
            for i in range(max(1, int(gauss(self.mean, self.sigma)))):
                creationId = self.creation_id
                endtime = self.meanEndTime + interval
                # endtime = max(1, int(gauss(self.meanEndTime, self.sigmaEndTime))) + interval
                # sla = self.meanSLA, self.sigmaSLA
                sla = max(1, int(gauss(self.meanSLA, self.sigmaSLA)))
                typeID = creationId % (len(self.vm_conf))
                # typeID = randint(0,len(self.vm_conf)-1)
                Cpu = self.vm_conf['cpu'][typeID]
                Core = self.vm_conf['core'][typeID]
                Ram = self.vm_conf['ram'][typeID]
                # Disk_ = self.vm_conf['DiskSize'][typeID]
                Bw_Up = self.vm_conf['bwUp'][typeID]
                Bw_Down = self.vm_conf['bwDown'][typeID]

                vmlist.append((creationId, interval, sla, endtime, Cpu, Core, Ram, Bw_Down, Bw_Down))
                self.creation_id += 1
            self.createdVms += vmlist
            self.deployedVms += [False] * len(vmlist)
        data = {
            "CreationID": [],
            "Interval": [],
            "SLA": [],
            "EndTime": [],
            "Cpu": [],
            "Core": [],
            "Ram": [],
            "BwUp": [],
            "BwDown": []
        }
        for creationId, interval, sla, endtime, Cpu, Core, Ram, Bw_Up, Bw_Down in self.createdVms:
            data["CreationID"].append(creationId)
            data["Interval"].append(interval)
            data["SLA"].append(sla)
            data["EndTime"].append(endtime)
            data["Cpu"].append(Cpu)
            data["Core"].append(Core)
            data["Ram"].append(Ram)
            data["BwUp"].append(Bw_Up)
            data["BwDown"].append(Bw_Down)
        df = pd.DataFrame(data=data)
        df.to_csv('envs/workload/scenarios/scenario2.csv', index=False)

def genScenario(meanNumVms, sigmaNumVms, meanEndTime=20, sigmaEndTime=200, meanSLA=20, sigmaSLA=3):
    """
    Generates a scenario with the given parameters.
    """
    workload = Bitbrain(meanNumVms, sigmaNumVms, meanEndTime, sigmaEndTime, meanSLA, sigmaSLA)
    workload.generateScenario(ep=2000)

genScenario(meanNumVms=2, sigmaNumVms=1, meanEndTime=20, sigmaEndTime=200, meanSLA=20, sigmaSLA=3)
    # genScenario(meanNumVms=10, sigmaNumVms=5, meanEndTime=20, sigmaEndTime=200, meanSLA=20, sigmaSLA=3)
