from IPSM import IPSM

class IPSMBitbrain(IPSM):
    def __init__(self, ips_list, max_ips, duration, SLA):
        super().__init__()
        self.ips_list = ips_list
        self.max_ips = max_ips
        self.SLA = SLA
        self.duration = duration
        self.completedInstructions = 0
        self.totalInstructions = 0

    def getIPS(self):
        pass