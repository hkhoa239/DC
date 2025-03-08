from .PM import PM

class PMConstant(PM):
    def __init__(self, idle=10, max=1000):
        super().__init__()
        self.idle = idle
        self.max = max
        # self.powerlist = [constant] * 11

    def power(self, utilization):
        return self.idle + (self.max - self.idle)*utilization
        # return self.constant