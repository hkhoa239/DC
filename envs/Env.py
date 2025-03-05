from .workload.BitbrainWorkload import BWGD
# from workload.BitbrainWorkload import BWGD
from datacenter.BitbrainDC import BitbrainDC
import gymnasium as gym
from Monitor import Monitor
from container.Container import Container
from .host.Host import Host
import time
import numpy as np

class Env():
    def __init__(self, TotalPower=10000, RouterBw=10000, ContainerLimit=10, IntervalTime=1, HostLimit=2, Monitor=Monitor()):
        self.totalpower = TotalPower
        self.totalbw = RouterBw
        self.hostlimit = HostLimit
        self.containerlimit = ContainerLimit
        self.intervaltime = IntervalTime
        self.curr_step = 0
        self.workload = BWGD(
            meanNumContainers=1,
            sigmaNumContainers=0.2
        )

        self.datacenter = BitbrainDC(num_hosts=HostLimit)
        self.hostlist = []

        self.containerlist = []
        self.inactivecontainers = []

        self.monitor = Monitor

        self.action_space = gym.spaces.Discrete(HostLimit)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(ContainerLimit*3+9,HostLimit)
        )

        self.initialized_flag = False
        self.interval = 0
        self.total_respone_time = 0
        self.total_completed_task = 0

        self.deployed = None

        self.resWeighted = 0.5  
        self.powWeighted = 0.5   
        self.historical_response_time = [1]
        self.historical_power_consumption = [1]    

    def reset(self, seed=42):
        if not self.initialized_flag:
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
            return state, info

            

    def getContainerByID(self, containerId):
        return self.containerlist[containerId]
    
    def getContainerByCID(self, creationId):
        for c in self.containerlist + self.inactiveContainers:
            if c and c.creationId == creationId:
                return c
            
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
        self.containerlist.append(container)
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

    # def allocateInit(self, action):
    #     migrations = []
    #     routerBwToEach = self.totalbw / len(action)

    # def addContainer(self, CreationID, CreationInterval, IPSModel, RAMModel, DiskModel):
    #     for i,c in enumerate(self.containerlist):
    #         if c == None or not c.active:
    #             container = Container(i, CreationID, CreationInterval, IPSModel, RAMModel, DiskModel, self, hostID=-1)
    #             self.containerlist[i] = container
    #             return container

    #addContainer base on host that using less power
    def addContainer(self, CreationID, CreationInterval, IPSModel, RAMModel, DiskModel):
        for i,c in enumerate(self.containerlist):
            if c == None or not c.active:
                container = Container(i, CreationID, CreationInterval, IPSModel, RAMModel, DiskModel, self, HostID=-1)
                self.containerlist[i] = container
                return container
            


    def addContainers(self, newContainerList):
        destroyed = self.destroyCompletedContainers()
        print("COMPLETED: ", [c.get_info() for c in destroyed])
        self.deployed = self.addContainerList(newContainerList)
        return self.deployed, destroyed
    

    # def addContainerList(self, containerInfoList):
    #     deployed = container

    def destroyCompletedContainers(self):
        destroyed = []
        for i,container in enumerate(self.containerlist):
            if container and container.getBaseIPS() == 0:
                self.total_respone_time += container.waitingTime + container.totalMigrationTime + container.totalExecTime
                self.total_completed_task += 1
                container.destroy()
                self.containerlist[i] = None
                self.inactivecontainers.append(container)
                destroyed.append(container)

        return destroyed
    


    def step(self, action):
        routerBwToEach = self.totalbw / len(action) if len(action) > 0 else self.totalbw
        migrations = []
        containerIDsAllocated = []
        print("EXECUTING step: ", self.curr_step)
        for (cid, hid) in action:
            print(f"{cid} ----------------- {hid}")
            time.sleep(1)
            if (self.containerlist[cid] == None):
                print("No Container: ", cid)
                break
            # chua giai quyet truong hop currentHost = -1
            container = self.getContainerByID(cid)
            currentHostID = self.getContainerByID(cid).getHostID()
            targetHost = self.getHostByID(hid)
            if (currentHostID == -1):
                
                migrationToNum = len(self.monitor.getMigrationToHost(hid, action=action))

                allocbw = min(targetHost.bwCapacity.downlink / migrationToNum, routerBwToEach)

                if hid != self.containerlist[cid].hostid and self.checkIfPossible(cid, hid):
                    migrations.append((cid, hid))
                    container.allocateAndExecute(hid, allocbw)
                    containerIDsAllocated.append(cid)
                elif not self.checkIfPossible(cid, hid):
                    print(f"Host {hid} is not enough resource.")
            
            else:
                currentHost = self.getHostByID(currentHostID)
            
                migrateFromNum = len(self.monitor.getMigrationFromHost(currentHostID, action))
                migrationToNum = len(self.monitor.getMigrationToHost(hid, action=action))

                allocbw = min(targetHost.bwCapacity.downlink / migrationToNum, currentHost.bwCapacity.uplink / migrateFromNum, routerBwToEach)

            if hid != self.containerlist[cid].hostid and self.checkIfPossible(cid, hid):
                migrations.append((cid, hid))
                container.allocateAndExecute(hid, allocbw)
                containerIDsAllocated.append(cid)
            elif not self.checkIfPossible(cid, hid):
                print(f"Host {hid} is not enough resource.")

        for cid in range(self.containerlimit):
            if self.containerlist[cid] and self.containerlist[cid].getHostID() != -1:
                if not self.checkIfPossible(cid, self.containerlist[cid].getHostID()):
                    self.containerlist[cid].hostid = -1

        for cid in range(self.containerlimit):
            if self.containerlist[cid] and self.containerlist[cid].hostid == -1:
                self.containerlist[cid].waitingTime += self.intervaltime

        for i,c in enumerate(self.containerlist):
            if c and i not in containerIDsAllocated and c.hostid != -1:
                migrations.append((c.id, c.hostid))
                c.execute(0)


        self.workload.updateDeployedContainers(self.getCreationIDs(migrations, self.deployed))


        # can chinh sua ham addContainers() -> nen co priority
        self.interval += self.intervaltime
        newinfoscontainer = self.workload.generateNewContainers(self.interval)

        self.addContainers(newinfoscontainer)

        self.curr_step += 1
        info = self.get_info()
        state = self.get_state()
        reward = self.calc_rew()
        self.update_weights()
        return state, reward, False, info              


    def get_info(self):
        hostl = []
        containerl = []

        for host in self.hostlist:
            hostl.append(host.get_info())

        for container in self.containerlist:
            if container:
                containerl.append(container.get_info())

        return {
            "hosts": hostl,
            "containers": containerl
        }
    
    def get_container_usage(self, host):
        usage = []
        for i,container in enumerate(self.containerlist):
            if container:
                IPSUse = container.getBaseIPS() / host.ipsCapacity
                rs, _, _ = container.getRAM()
                rsa, _, _ = host.getRAMAvailable()
                RamSizeUse = rs / rsa
                ds, _, _ = container.getDisk()
                dsa, _, _ = host.getDiskAvailable()
                DiskSizeUse = ds / dsa
                usage.extend([IPSUse, RamSizeUse, DiskSizeUse])
            else:
                usage.extend([0,0,0])
        return usage

    def get_state(self):
        state = []
        for i in range(self.hostlimit):
            host = self.getHostByID(i)
            host_state = host.get_state()
            container_usage = self.get_container_usage(host)
            host_state = [1] + host_state + container_usage

            # print(len(host_state))
            state.append(host_state)
        state = np.array(state).T
        return state

    def filter_action(self, action):
        decision = []
        for i, c in enumerate(self.containerlist):
            if c and c.getHostID() != action and c.getHostID() == -1:
                decision.append((c.id, action))

        return decision
    
    def calc_rew(self):
        totalPowerConsumption = sum(host.calculateHostPowerConsumption() for host in self.hostlist)
        self.historical_response_time.append(self.total_respone_time)
        self.historical_power_consumption.append(totalPowerConsumption)
        
        max_response_time = max(self.historical_response_time)
        max_power_consumption = max(self.historical_power_consumption)


        norm_response_time  = self.total_respone_time / max_response_time
        norm_power_consumption  = totalPowerConsumption / max_power_consumption

        r_T = 1 / (1 + norm_response_time)  
        r_E = 1 / (1 + norm_power_consumption)  
        
        # Reward tổng hợp dựa trên trọng số
        reward = self.resWeighted * r_T + self.powWeighted * r_E
        return reward        
        
    def update_weights(self):
        if self.total_respone_time > 1.5:  # Nếu trễ cao, tăng trọng số độ trễ
            self.resWeighted = min(1.0, self.resWeighted + 0.1)
            self.powWeighted = max(0.0, self.powWeighted - 0.1)
        elif sum(host.calculateHostPowerConsumption() for host in self.hostlist) > 5000:  # Nếu năng lượng cao, tăng trọng số năng lượng
            self.resWeighted = max(0.0, self.resWeighted - 0.1)
            self.powWeighted = min(1.0, self.powWeighted + 0.1)
                
    def getPowerConsumption(self):
        totalPower = 0
        hosts = set(container.getHost() for container in self.containerlist if container is not None) 
        
        for host in hosts:
            containers = [c for c in self.containerlist if c is not None and c.getHost() == host]
            if not containers:
                continue  

            CPU_utilization = sum(c.getCPU() for c in containers) / 100
            totalPower += host.getPowerCPU(min(1, CPU_utilization)) 

        return totalPower