from .workload.BitbrainWorkload import BWGD
# from workload.BitbrainWorkload import BWGD
from datacenter.BitbrainDC import BitbrainDC
import gymnasium as gym
from Monitor import Monitor
from container.Container import Container
from .host.Host import Host
import time

class Env():
    def __init__(self, TotalPower=10000, RouterBw=10000, ContainerLimit=10, IntervalTime=1, HostLimit=2, Monitor=Monitor()):
        self.totalpower = TotalPower
        self.totalbw = RouterBw
        self.hostlimit = HostLimit
        self.containerlimit = ContainerLimit
        self.intervaltime = IntervalTime

        self.workload = BWGD(
            meanNumContainers=1,
            sigmaNumContainers=0.2
        )

        self.datacenter = BitbrainDC(num_hosts=HostLimit)
        self.hostlist = []

        self.containerlist = []
        self.inactivecontainers = []

        self.monitor = Monitor

        self.action_space = None
        self.observation_spaces = None

        self.initialized_flag = False
        self.interval = 0

        self.deployed = None
        

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
            
            info = self.get_info()
            return info

            

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
                container.destroy()
                self.containerlist[i] = None
                self.inactivecontainers.append(container)
                destroyed.append(container)

        return destroyed
    


    def step(self, action):
        routerBwToEach = self.totalbw / len(action) if len(action) > 0 else self.totalbw
        migrations = []
        containerIDsAllocated = []
        print("EXECUTING step: ", self.interval)
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
            if self.containerlist[cid] and self.containerlist[cid].hostid == -1:
                self.containerlist[cid] = None

        for i,c in enumerate(self.containerlist):
            if c and i not in containerIDsAllocated and c.hostid != -1:
                migrations.append((c.id, c.hostid))
                c.execute(0)


        self.workload.updateDeployedContainers(self.getCreationIDs(migrations, self.deployed))


        # can chinh sua ham addContainers() -> nen co priority
        self.interval += 1
        newinfoscontainer = self.workload.generateNewContainers(self.interval)

        self.addContainers(newinfoscontainer)

        info = self.get_info()

        return info              


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

    def get_obs(self):
        pass

    def calc_rew(self):
        pass
