
class Container():
    # IPS = ips requirement
	# RAM = ram requirement in MB
	# Size = container size in MB
    

    def __init__(self, ID, creationId, creationInterval, IPSModel, RAMModel, DiskModel, Environment, HostID=-1):
        self.id = ID
        self.creationID = creationId
        self.hostid = HostID
        self.env = Environment
        self.createAt = creationInterval
        self.startAt = self.env.interval
        self.totalExecTime = 0
        self.totalMigrationTime = 0
        self.active = True
        self.destroyAt = -1
        self.lastContainerSize = 0

        self.ipsmodel = IPSModel
        self.ipsmodel.allocContainer(self)
        self.sla = self.ipsmodel.SLA
        self.rammodel = RAMModel
        self.rammodel.allocContainer(self)
        self.diskmodel = DiskModel
        self.diskmodel.allocContainer(self)


    def getBaseIPS(self):
        return self.ipsmodel.getIPS()

    def getApparentIPS(self):
        if self.hostid == -1:
            return self.ipsmodel.getMaxIPS()

    def getRAM(self):
        rsize, rread, rwrirte = self.rammodel.ram()
        self.lastContainerSize = rsize
        return rsize, rread, rwrirte
    
    def getDisk(self):
        return self.diskmodel.disk()
    
    def getContainerSize(self):
        if self.lastContainerSize == 0: 
            self.getRam()
        return self.lastContainerSize
    
    def getHostID(self):
        return self.hostid
    
    def getHost(self):
        return self.env.getHostByID(self.hostid)

    def allocate(self, hostId, allocBw):
        lastMigrationTime = 0
        if self.hostid != hostId:
            lastMigrationTime += self.getContainerSize() / allocBw
            lastMigrationTime += abs(self.env.hostlist[self.hostid].latency - self.env.hostlist[hostId].latency)
        self.hostid = hostId
        return lastMigrationTime
    
    def execute(self, lastMigrationTime):
        assert self.hostid != -1
        self.totalMigrationTime += lastMigrationTime
        execTime = self.env.intervaltime - lastMigrationTime
        apparentIPS = self.getApparentIPS()
        requiredExecTime = (self.ipsmodel.totalInstructions - self.ipsmodel.completedInstruction) / apparentIPS if apparentIPS else 0
        self.ipsmodel.completedInstructions += apparentIPS * min (execTime, requiredExecTime)

    def allocateAndExecute(self, hostId, allocBw):
        self.execute(lastMigrationTime=self.allocate(hostId, allocBw))

    def destroy(self):
        self.destroyAt = self.env.interval
        self.hostid = -1
        self.active = False
