class Monitor():
    def __inti__(self, env=None):
        self.env = env

    def setEvironment(self, env):
        self.env = env

    def getMigrationFromHost(self, HostId, action):
        containerIDs = []
        for cid, _ in action:
            hid = self.env.getContainerByID(cid).getHostID()
            if hid == HostId:
                containerIDs.append(cid)

        return containerIDs
    
    def getMigrationToHost(self, hostId, action):
        containerIDs = []
        for (cid, hid) in action:
            if hid == hostId:
                containerIDs.append(cid)

        return containerIDs