from metrics.powermodels.PMLinear import PMLinear
class Host():
    def __init__(self, id, cpu, core, ram, bwup, bwdown, powermodel:PMLinear, environment):
        self.id = id
        self.cpu = cpu
        self.core = core
        self.ram = ram
        self.bwup = bwup
        self.bwdown = bwdown
        self.env = environment
        self.powermodel = powermodel

        self.ramUsed = 0
        self.coreUsed = 0
        self.vmlist = []

    def getBwUp(self):
        return self.bwup - self.bwupUsed
    
    def getBwDown(self):
        return self.bwdown - self.bwdownUsed

    def getRamAv(self):
        return self.ram - self.ramUsed
    
    def getCoreAv(self):
        return self.core - self.coreUsed
    
    def allocate(self, vm):
        self.vmlist.append(vm)
        self.coreUsed += vm.core
        self.ramUsed += vm.ram

    def destroyVm(self, vm):
        self.coreUsed -= vm.core
        self.ramUsed -= vm.ram
        self.vmlist.remove(vm)
