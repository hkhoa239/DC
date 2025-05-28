from .Workload import Workload
import pandas as pd

class Scenario(Workload):
    def __init__(self, numScenario=1):
        super().__init__()
        self.data = pd.read_csv(f'envs/workload/scenarios/scenario{numScenario}.csv')

    def generateNewVms(self, interval):
        vmlist = []
        data_interval = self.data[self.data['Interval'] == interval]
        for index, row in data_interval.iterrows():
            creationId = row['CreationID']
            sla = row['SLA']
            endtime = row['EndTime']
            Cpu = row['Cpu']
            Core = row['Core']
            Ram = row['Ram']
            Bw_Up = row['BwUp']
            Bw_Down = row['BwDown']

            vmlist.append((creationId, interval, sla, endtime, Cpu, Core, Ram, Bw_Up, Bw_Down))
            self.creation_id += 1
        self.createdVms += vmlist
        self.deployedVms += [False] * len(vmlist)
        return vmlist


    def destroyVms(self, interval):
        cids = []
        after1 = []
        after2 = []
        self.vmCopy = self.createdVms.copy()
        for i, (creationId, interv, sla, endtime, Cpu, Core, Ram, Bw_Down, Bw_Down) in enumerate(self.vmCopy):
            if endtime <= interval:
                cids.append(creationId)
        return cids