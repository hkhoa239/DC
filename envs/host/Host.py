# from Disk import Disk
# from RAM import RAM
# from Bandwitdh import Bandwidth
class Host():
    # IPS = Million Instructions per second capacity 
	# RAM = Ram in MB capacity
	# Disk = Disk characteristics capacity
	# Bw = Bandwidth characteristics capacity

    def __init__(self, ID, IPS, RAM, Disk, Bw, Latency, Powermodel, Environment):
        self.id = ID
        self.ipsCapacity = IPS
        self.ramCapacity = RAM
        self.diskCapacity = Disk
        self.bwCapacity = Bw
        self.latency = Latency
        self.powermodel = Powermodel
        self.powermodel.allocHost(self)
        self.powermodel.host = self
        self.env = Environment
        
    def getPower(self):
        return self.powermodel.power()
    
    def getPowerFromIPS(self, ips):
        # Ensure CPU utilization is limited at 100%
        return self.powermodel.powerFromCPU(min(100, 100*(ips/self.ipsCapacity)))

    def getCPU(self):
        ips = self.getApparentIPS()
        return 100 * (ips / self.ipsCapacity)

    def getCPUAvailable(self):
        ips = self.getIPSAvailable()
        return 100 * (ips / self.ipsCapacity)
        
    def getBaseIPS(self):
        #Cal sum base IPS from all containers assigned to the host
        ips = sum([self.env.getContainerByID(containerID).getBaseIPS() for containerID in self.env.getContainersOfHost(self.id)])
        return ips
    
    def getApparentIPS(self):
        ips = sum([self.env.getContainerByID(containerID).getApparentIPS() for containerID in self.env.getContainersOfHost(self.id)])
        return int(ips)
    
    def getIPSAvailable(self):
        #Cal remaining IPS capacity that can be allocated to new containers 
        return self.ipsCapacity - self.getBaseIPS()
    
    def getCurrentRAM(self):
        size, read, write = 0, 0, 0
        containers = self.env.getContainersOfHost(self.id)
        for containerID in containers:
            s, r, w = self.env.getContainerByID(containerID).getRAM()
            size += s
            read += r
            write += w
        return size, read, write
    
    def getRAMAvailable(self):
        size, read, write = self.getCurrentRAM()
        return self.ramCapacity.size - size, self.ramCapacity.read - read, self.ramCapacity.write - write  

    def getCurrentDisk(self):
        size, read, write = 0, 0, 0
        containers = self.env.getContainersOfHost(self.id)
        for containerID in containers:
            s, r, w = self.env.getContainerByID(containerID).getDisk()
            size += s
            read += r
            write += w
        return size, read, write
    
    def getDiskAvailable(self):
        size, read, write = self.getCurrentDisk()
        return self.diskCapacity.size - size, self.diskCapacity.read - read, self.diskCapacity.write - write  

    def canAllocateContainer(self, container):
        if not container:
            return False

        ipsreq = container.getBaseIPS()
        ramreq, _, _ = container.getRAM()
        diskreq, _, _ = container.getDisk()
        
        if ipsreq <= self.getIPSAvailable() and ramreq <= self.getRAMAvailable()[0] and diskreq <= self.getDiskAvailable()[0]:
            container.hostid = self.id
            return True
        return False
        
    def calculateHostPowerConsumption(self):
        containers = self.env.getContainersOfHost(self.id)
        CPU_utilization = sum(c.getCPU() for c in containers) / self.getCPUAvailable()
        print("CPU_util", CPU_utilization)
        totalPower = self.getPowerCPU(CPU_utilization)
        return totalPower