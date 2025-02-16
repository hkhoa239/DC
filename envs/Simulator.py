from host.Host import Host
from container.Container import Container

class Simulator():
    # Total power in watt
	# Total Router Bw
	# Interval Time in seconds

    def __init__(self, TotalPower, RouterBw, Scheduler, ContainerLimit, IntervalTime, hostinit):
        self.totalpower = TotalPower
        self.totalbw = RouterBw
        self.hostlimit = len(hostinit)
        self.scheduler = Scheduler
        self.scheduler.setEnvironment(self)
        self.containerlimit = ContainerLimit
        self.hostlist = []
        self.containerlist = []
        self.intervaltime = IntervalTime
        self.interval = 0
        self.inactiveContainers = []
        self.stats = None
        self.addHostlistInit(hostinit)

    def addHostInit(self, IPS, RAM, Disk, Bw, Latency, PowerModel):
        assert len(self.hostlist) < self.hostlimit
        host = Host(ID=len(self.hostlist), IPS=IPS, RAM=RAM, Disk=Disk, Bw=Bw, Latency=Latency, Powermodel=PowerModel, Environment=self)
        self.hostlist.append(host)

    def addHostListInit(self, hostList):
        assert len(hostList) == self.hostlimit
        for IPS, RAM, Disk, Bw, Latency, Powermodel in hostList:
            self.addHostInit(IPS, RAM, Disk, Bw, Latency, Powermodel)

    def addContainerInit(self, CreationID, CreationInterval, IPSModel, RAMModel, DiskModel):
        container = Container(ID=len(self.container), creationId=CreationID, creationInterval=CreationInterval, IPSModel=IPSModel, RAMModel=RAMModel, DiskModel=DiskModel, Environment=self, HostID=1)

        self.containerlist.append(container)
        return container


    def addContainerListInit(self, containerInfoList):
        deployed = containerInfoList[:min(len(containerInfoList), self.containerlimit-self.getNumActiveContainers())]
        deployedContainers = []
        for CreationID, CreationInterval, IPSModel, RAMModel, DiskModel in deployed:
            dep = self.addContainerInit(CreationID, CreationInterval, IPSModel, RAMModel, DiskModel)
            deployedContainers.append(dep)
        self.containerlist += [None] * (self.containerlimit - len(self.containerlist))
        return [container.id for container in deployedContainers]
    
    def addContainer(self, CreationID, CreationInterval, IPSModel, RAMModel, DiskModel):
        pass

    def addContainerList(self, containerInfoList):
        pass

    def getContainerOfHost(self, hostID):
        containers = []

        for container in self.containerlist:
            if container and container.hostid == hostID:
                containers.append(container.id)

        return containers
    
    def getContainerByCID(self, creationID):
        pass

    def getHostByID(self, hostID):
        return self.hostlist[hostID]
    
    def getCreationIDs(self, migrations, containerIDS):
        pass

    


