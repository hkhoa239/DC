from .workload.BitbrainWorkload import BWGD
# from workload.BitbrainWorkload import BWGD
from datacenter.BitbrainDC import BitbrainDC
import gymnasium as gym
from Monitor import Monitor
from container.Container import Container
from .host.Host import Host
import time
import numpy as np
import json

class Env():
    def __init__(self, TotalPower=1000, RouterBw=10000, ContainerLimit=10, IntervalTime=1, HostLimit=3, meanJ=5, sigmaJ=1, Monitor=Monitor()):
        self.totalpower = TotalPower
        self.totalbw = RouterBw
        self.hostlimit = HostLimit
        self.containerlimit = ContainerLimit
        self.intervaltime = IntervalTime
        self.workload = BWGD(
            meanNumContainers=meanJ,
            sigmaNumContainers=sigmaJ
        )

        self.datacenter = BitbrainDC(num_hosts=HostLimit)

        self.monitor = Monitor

        self.action_space = gym.spaces.Discrete(HostLimit)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(ContainerLimit*3+9,HostLimit)
        )

        self.initialized_flag = False
           

    def reset(self, seed=42):
        if not self.initialized_flag:
            self.curr_step = 0
            self.interval = 0
            self.total_completed_task = 0
            self.steppower = 0
            self.deployed = None

            self.resWeighted = 0.5  
            self.powWeighted = 0.5   
            self.historical_response_time = []
            self.historical_power_consumption = [] 
            self.hostlist = []

            self.containerlist = []
            self.fifo = []
            self.dropped = 0

            containerinfosinit = self.workload.generateNewContainers(interval=self.interval)
            
            self.monitor.setEvironment(self)
            self.deployed = self.addContainersInit(containerinfosinit)
            self.initialized_flag = True

            hostinfo = self.datacenter.generateHosts()
            # create host 
            for i, (IPS, RAM, Disk, Bw, Latency, Powermodel) in enumerate(hostinfo):
                host = Host(
                    ID = i,
                    IPS = IPS,
                    RAM = RAM,
                    Disk = Disk,
                    Bw = Bw,
                    Latency = Latency,
                    Powermodel = Powermodel,
                    Environment = self
                )
                self.hostlist.append(host)
            
            state = self.get_state()
            info = self.get_info()
            # self.save_config(path="simdata/host.json")
            return state, info

            

    def getContainerByID(self, containerId):
        return self.containerlist[containerId]
    
    # def getContainerByCID(self, creationId):
    #     for c in self.containerlist + self.inactiveContainers:
    #         if c and c.creationId == creationId:
    #             return c
            
    def getContainersOfHost(self, hostId):
        containers = []
        for container in self.containerlist:
            if container and container.hostid == hostId:
                containers.append(container.id)

        return containers

    def getHostByID(self, hostId):
        return self.hostlist[hostId]
    
    def getNumActiveContainers(self):
        num = 0
        for container in self.containerlist:
            if container and container.active: 
                num += 1
        return num


    def getCreationIDs(self, migrations, containerIDs):
        creationIDs = []
        for action in migrations:
            if action[0] in containerIDs: 
                creationIDs.append(self.containerlist[action[0]].creationID)
        return creationIDs

    def checkIfPossible(self, containerId, hostId):
        container = self.containerlist[containerId]
        host = self.hostlist[hostId]

        ipsreq = container.getBaseIPS()
        ramsizereq, ramreadreq, ramwritereq = container.getRAM()

        disksizereq, diskreadreq, diskwritereq = container.getDisk()

        ipsav = host.getIPSAvailable()
        ramsizeav, ramreadav, ramwriteav = host.getRAMAvailable()

        disksizeav, diskreadav, diskwriteav = host.getDiskAvailable()

        return ipsreq <= ipsav and ramsizereq <= ramsizeav and ramreadreq <= ramreadav and ramwritereq <= ramwriteav and disksizereq <= disksizeav and diskreadreq <= diskreadav and diskwritereq <= diskwriteav 

    def addContainersInit(self, containerInfoListInit):
        # self.interval += 1
        deployed = self.addContainerListInit(containerInfoListInit)
        return deployed

    def addContainerListInit(self, containerInfoList):
        deployed = containerInfoList[:min(len(containerInfoList), self.containerlimit - self.getNumActiveContainers())]
        deployedContainers = []
        for CreationID, CreationInterval, IPSModel, RAMModel, DiskModel in deployed:
            dep = self.addContainerInit(
                CreationID, CreationInterval, IPSModel, RAMModel, DiskModel
            )
            deployedContainers.append(dep)
        self.containerlist += [None] * (self.containerlimit - len(self.containerlist))
        return [container.id for container in deployedContainers]


    def addContainerInit(self, CreationID, creationInterval, IPSModel, RAMModel, DiskModel):
        container = Container(
            ID=len(self.containerlist),
            creationId=CreationID,
            creationInterval=creationInterval,
            IPSModel=IPSModel,
            RAMModel=RAMModel,
            DiskModel=DiskModel,
            Environment=self,
            HostID = -1
        )
        # heapkey = container.ipsmodel.getIPSRemain() / container.getTimeRemain()
        # self.workload.heap.insert(key=1/heapkey, value=container.id)
        self.containerlist.append(container)
        self.fifo.append(container.id)
        return container
    
    def addContainerList(self, containerInfoList):
        deployed = containerInfoList[:min(len(containerInfoList), self.containerlimit - self.getNumActiveContainers())]
        deployedContainers = []
        for CreationID, CreationInterval, IPSModel, RAMModel, DiskModel in deployed:
            dep = self.addContainer(
                CreationID, CreationInterval, IPSModel, RAMModel, DiskModel
            )
            deployedContainers.append(dep)
        
        return [container.id for container in deployedContainers]

    #addContainer base on host that using less power
    def addContainer(self, CreationID, CreationInterval, IPSModel, RAMModel, DiskModel):
        for i,c in enumerate(self.containerlist):
            if c == None or not c.active:
                container = Container(i, CreationID, CreationInterval, IPSModel, RAMModel, DiskModel, self, HostID=-1)
                self.containerlist[i] = container
                self.fifo.append(container.id)
                return container
            


    def addContainers(self, newContainerList):
        self.deployed = self.addContainerList(newContainerList)
        return self.deployed
    

    # def addContainerList(self, containerInfoList):
    #     deployed = container

    def destroyCompletedContainers(self):
        destroyed = []
        for i,container in enumerate(self.containerlist):
            if container and container.getBaseIPS() == 0:
                if (len(self.historical_response_time)>50):
                    self.historical_response_time = self.historical_response_time[-50:]
                self.historical_response_time.append(container.waitingTime + container.totalMigrationTime + container.totalExecTime)
                
                self.total_completed_task += 1
                container.destroy()
                self.containerlist[i] = None
                self.fifo.remove(i)
                # self.inactivecontainers.append(container)
                destroyed.append(container)

        # print("COMPELTED: ", [container.get_info() for container in destroyed])
        return destroyed
    


    def step(self, action):
        routerBwToEach = self.totalbw / len(action) if len(action) > 0 else self.totalbw
        self.steppower = 0
        migrations = []
        containerIDsAllocated = []


        for i,c in enumerate(self.containerlist):
            if c and c.sla < self.interval + self.intervaltime - c.createAt:
                self.containerlist[i] = None
                self.fifo.remove(c.id)
                self.dropped += 1

        for cid in range(self.containerlimit):
            if self.containerlist[cid] and self.containerlist[cid].getHostID() != -1:
                if not self.checkIfPossible(cid, self.containerlist[cid].getHostID()):
                    self.containerlist[cid].hostid = -1

        for (cid, hid) in action:
            # print(f"CID {cid} - HID {hid}")
            if (self.containerlist[cid] != None):
                container = self.getContainerByID(cid)
                currentHostID = self.getContainerByID(cid).getHostID()
                targetHost = self.getHostByID(hid)
                if (currentHostID == -1):
                    
                    migrationToNum = len(self.monitor.getMigrationToHost(hid, action=action))

                    allocbw = min(targetHost.bwCapacity.downlink / migrationToNum, routerBwToEach)

                    if hid != self.containerlist[cid].hostid and self.checkIfPossible(cid, hid):
                        migrations.append((cid, hid))
                        container.allocate(hid, allocbw)
                        containerIDsAllocated.append(cid)
                    # else:
                        # print(f"Host {hid} is not enough resource.")
                
                else:
                    currentHost = self.getHostByID(currentHostID)
                
                    migrateFromNum = len(self.monitor.getMigrationFromHost(currentHostID, action))
                    migrationToNum = len(self.monitor.getMigrationToHost(hid, action=action))

                    allocbw = min(targetHost.bwCapacity.downlink / migrationToNum, currentHost.bwCapacity.uplink / migrateFromNum, routerBwToEach)

                    if hid != self.containerlist[cid].hostid and self.checkIfPossible(cid, hid):
                        migrations.append((cid, hid))
                        container.allocate(hid, allocbw)
                        containerIDsAllocated.append(cid)
                    # else:
                        # print(f"Host {hid} is not enough resource.")

        for c, h  in migrations:
            container = self.containerlist[c]
            container.execute()

        for cid in range(self.containerlimit):
            if self.containerlist[cid] and self.containerlist[cid].hostid == -1:
                self.containerlist[cid].waitingTime += self.intervaltime

        for i,c in enumerate(self.containerlist):
            if c and i not in containerIDsAllocated and c.hostid != -1:
                migrations.append((c.id, c.hostid))
                c.execute()

        # self.workload.updateDeployedContainers(self.getCreationIDs(migrations, self.deployed))
        destroyed = self.destroyCompletedContainers()

        self.interval += self.intervaltime

        reward = self.calc_rew()
        
        newinfoscontainer = self.workload.generateNewContainers(self.interval)

        self.addContainers(newinfoscontainer)

        self.curr_step += 1
        state = self.get_state()
        info = self.get_info()
        # self.save_data(data=info, path='simdata/data.json')
        return state, reward, False, info              


    def get_info(self):
        hostl = []
        containerl = []

        for host in self.hostlist:
            hostl.append(host.get_info())

        # for container in self.containerlist:
        #     if container:
        #         containerl.append(container.get_info())

        return {
            "step": self.curr_step-1,
            "hosts": hostl,
            "completed": self.total_completed_task,
            "dropped": self.dropped,
            # "containers": containerl,
            "power_consumption": self.steppower,
            "respone_time_history": self.historical_response_time,
            "fifo": len(self.fifo)
        }
    
    def get_container_usage(self, host):
        usage = []
        for i in self.fifo:
            container = self.containerlist[i]
            IPSUse = container.getBaseIPS() / host.ipsCapacity
            rs, _, _ = container.getRAM()
            rsa, _, _ = host.getRAMAvailable()
            RamSizeUse = rs / rsa
            ds, _, _ = container.getDisk()
            dsa, _, _ = host.getDiskAvailable()
            DiskSizeUse = ds / dsa
            usage.extend([IPSUse, RamSizeUse, DiskSizeUse])
        
        usage.extend([0,0,0] * (self.containerlimit - len(self.fifo)))
        return usage

    def get_state(self):
        state = []
        for i in range(self.hostlimit):
            host = self.getHostByID(i)
            host_state = host.get_state()
            container_usage = self.get_container_usage(host)
            host_state = [1] + host_state + container_usage

            state.append(host_state)
        state = np.array(state).T
        return state

    def filter_action(self, action):
        decision = []
        
        for i in self.fifo:
            c = self.containerlist[i]
            if c and c.getHostID() != action and c.getHostID() == -1:
                decision.append((c.id, action))

        return decision
    

    def calc_rew(self):
        self.steppower = sum(host.calculateHostPowerConsumption() for host in self.hostlist)
        
        self.historical_power_consumption.append(self.steppower)
        

        norm_response_time  = ( sum(self.historical_response_time) / len(self.historical_response_time) ) / max(self.historical_response_time) if len(self.historical_response_time) > 0 and max(self.historical_response_time) > 0 else 0
        norm_power_consumption  = self.steppower / self.totalpower

        r_T = -norm_response_time
        r_E = -norm_power_consumption
        
        reward = self.resWeighted * r_T + self.powWeighted * r_E + (-self.dropped / self.total_completed_task if self.total_completed_task > 0 else 0)
        return reward        
        

    def save_config(self, path):
        hostl = []
        for i in self.hostlist:
            hostl.append({
                "ipsCap": i.ipsCapacity,
                "ramSizeCap": i.ramCapacity.size,
                "diskSizeCap": i.diskCapacity.size,
                # "bwCap": i.bwCapacity,
                # "latency": i.latency
            })
        try:
            with open(path, 'r') as json_file:
                existing_data = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = {'host': []}
        existing_data = hostl
        with open(path, 'w') as json_file:
            json.dump(existing_data, json_file, indent=4)

    def save_data(self, data, path):
        try:
            with open(path, 'r') as json_file:
                existing_data = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = {'random_data': []}
        existing_data['random_data'].append(data)

        with open(path, 'w') as json_file:
            json.dump(existing_data, json_file, indent=4)