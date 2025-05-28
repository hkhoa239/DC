# from .workload.Bitbrain import Bitbrain
from .workload.Scenario import Scenario
from .datacenter.DC import DC
from host.Host import Host
from Monitor import Monitor
from vm.VM import VM
import gymnasium as gym

class Env():
    def __init__(self, TotalPower=2000, RouterBw=10000, NumHost=21, VmLimit=1000, IntervalTime=1):
        self.totalpower = TotalPower
        self.totalbw = RouterBw
        self.intervaltime = IntervalTime
        self.interval = 0
        self.workload = Scenario(numScenario=1)
        # self.workload = Bitbrain(
        #     meanNumVms=1,
        #     sigmaNumVms=1
        # )
        self.datacenter = DC(
            num_hosts=NumHost, environment=self
        )

        self.hostlist : list[Host] = []
        self.vmlimit = VmLimit
        self.vmlist : list[VM] = [None] * VmLimit
        self.inactiveVmId: list[int] = []
        self.activeVmId: list[int] = []

        self.monitor = Monitor(environment=self)

        self.step_power = 0
        self.initialize_flag = False        

        self.action_space = gym.spaces.Discrete(NumHost)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=float('inf'),
            shape=(4,NumHost)
        )


    def reset(self, seed=42):
        if not self.initialize_flag:
            self.vmlist = [None] * self.vmlimit
            self.inactiveVmId = []
            self.activeVmId = []
            self.hostlist = []
            self.interval = 0
            self.step_power = 0
            self.workload.reset()

            self.initHost()
            self.addWorkloadList()
            self.updateMatrix()
            state = self.getState()
            info = self.getInfo()

        return state, info

    def getNumVms(self):
        num = 0
        for vm in self.vmlist:
            if vm:
                num+=1
        return num

    def addWorkloadList(self):
        vmInfoList = self.workload.generateNewVms(interval=self.interval)
        vmInfoList = vmInfoList[:min(len(vmInfoList), self.vmlimit-self.getNumVms())]
        deployedVms = []
        for creationid, creationinterval, sla, endtime, cpu, core, ram, bwup, bwdown in vmInfoList:
            dep = self.addWorkload(creationid, creationinterval, sla, endtime, cpu, core, ram, bwup, bwdown)
            deployedVms.append(dep)
        return [vm.id for vm in deployedVms]

    def addWorkload(self, creationid, creationinterval, sla, endtime, cpu, core, ram, bwup, bwdown):
        for i, vm in enumerate(self.vmlist):
            if vm == None:
                vm = VM(
                    id=i,
                    creationId=creationid,
                    creationInterval=creationinterval,
                    sla=sla,
                    endtime=endtime,
                    cpu=cpu,
                    core=core,
                    ram=ram,
                    bwup=bwup,
                    bwdown=bwdown,
                    environment=self
                )
                self.vmlist[i] = vm
                self.inactiveVmId.append(i)
                return vm


    def initHost(self):
        hostinfo = self.datacenter.generateHosts()
        for i, (cpu, core, ram, bwup, bwdown, power) in enumerate(hostinfo):
            host = Host(
                id=i,
                cpu=cpu,
                core=core,
                ram=ram,
                bwup=bwup,
                bwdown=bwdown,
                powermodel=power,
                environment=self
            )
            self.hostlist.append(host)

    def getVmByID(self, vid):
        for vm in self.vmlist:
            if vm and vm.id == vid:
                return vm

    def getHostByID(self, hid):
        for host in self.hostlist:
            if host.id == hid:
                return host
            
    def assignVmToHost(self, vm:VM, host:Host):
        host.allocate(vm)
        vm.updateHost(host)
        vm.handle(action=1)
        self.inactiveVmId.remove(vm.id)
        self.activeVmId.append(vm.id)
        self.workload.updateDeployedVms([vm.creationId])
        

    def checkPlacement(self, vm:VM, host:Host):
        if vm.cpu != host.cpu:
            return False
        if vm.core > host.getCoreAv() or vm.ram > host.getRamAv():
            return False
        return True

    def calcPowerConsumption(self):
        power = 0
        for host in self.hostlist:
            power += host.powermodel.power(host.coreUsed/host.core)
        return power

    def destroyVmByCID(self, cid):
        vmCopy = self.vmlist.copy()
        for vmid in self.activeVmId:
            if vmCopy[vmid].creationId == cid:
                vm = vmCopy[vmid]
                hostid = vm.hostid
                host = self.getHostByID(hostid)
                host.destroyVm(vm=vm)
                vm.handle(action=-1)

                self.vmlist[vmid] = None
                self.activeVmId.remove(vmid)
                return
            
        for vmid in self.inactiveVmId:
            if vmCopy[vmid].creationId == cid:
                self.vmlist[vmid] = None
                self.inactiveVmId.remove(vmid)
                return

    def handleDestroyVms(self):
        vmDestroyedByCIDs = self.workload.destroyVms(interval=self.interval)
        for cid in vmDestroyedByCIDs:
            self.destroyVmByCID(cid)

    def handleGenerateVms(self):
        self.addWorkloadList()
    

    def updateWorkload(self):
        self.handleDestroyVms()
        self.handleGenerateVms()

    def updateAfterSim(self):
        self.interval += self.intervaltime
        self.updateWorkload()
        self.updateMatrix()
        state = self.getState()
        return state


    def calcRew(self):
        return self.monitor.calcRew()

    def getState(self):
        return self.monitor.getState()

    def getInfo(self):
        info = {
            "Interval": self.interval,
            "Power_Consumption": self.step_power,
            "CPU_Utilization": [host[2] for host in self.monitor.hostMatrix],
            "Ram_Utilization": [host[3] for host in self.monitor.hostMatrix],
            "Inactivevmids": [vmid for vmid in self.inactiveVmId],
            "CPU_Imbalance": self.monitor.imCore,
            "Ram_Imbalance": self.monitor.imRam,
        }
        return info
    
    def getDone(self):
        return False

    def getHostMatrix(self):
        return self.monitor.hostMatrix
    
    def getVmMatrix(self):
        return self.monitor.vmMatrix

    def updateMatrix(self):
        self.monitor.createHostMatrix()
        self.monitor.createVmMatrix()

    def getSelectableVms(self):
        return self.inactiveVmId


    def step(self,decision):
        for (vid, hid) in decision:
            vm = self.getVmByID(vid)
            host = self.getHostByID(hid)

            if self.checkPlacement(vm, host):
                self.assignVmToHost(vm, host)


        self.step_power = self.calcPowerConsumption()
        self.updateMatrix()

        self.updateAfterSim()

        # Không cần model
        state = self.getState()
        rew = self.calcRew()
        info = self.getInfo()
        done = self.getDone()
        # print("SHAPE: ", state.shape)
        return state, rew, done, info


            