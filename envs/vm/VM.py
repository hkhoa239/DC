

class VM():
    def __init__(self, id, creationId, creationInterval, sla, endtime, cpu, core, ram, bwup, bwdown, environment, hostId=-1):
        self.id = id
        self.creationId = creationId
        self.cpu = cpu
        self.core = core
        self.ram = ram
        self.sla = sla
        self.hostid = hostId
        self.host = None
        self.env = environment
        self.createAt = creationInterval
        self.startAt = -1
        self.destroyAt = -1
        self.status = None
        self.sla = sla
        self.endTime = endtime
        self.bwup = bwup
        self.bwdown = bwdown

        self.acts = {
            "START": 1,
            "STOP": 0,
            "DESTROY": -1
        }

    def handle(self, action):
        if action == self.acts['START']:
            self.startAt = self.env.interval
            self.status = action

        elif action == self.acts['STOP']:
            self.status = action
        
        elif action == self.acts['DESTROY']:
            self.host = None
            self.hostid = -1
            self.status = action
            self.destroyAt = self.env.interval

    def updateHost(self, host):
        self.host = host
        self.hostid = host.id