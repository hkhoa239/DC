import math

class PM():
    def __init__(self, host=None):
        self.host = host
        # we haven't has powerlist from any power model

    def allocHost(self, host):
        self.host = host

    def powerFormCPU(self, cpu):
        """_summary_
        Params:
            cpu (float): CPU utilization percentage (from 0 to 100)
        Returns:
            float: power consumption  
        """
        index = math.floor(cpu/10)
        left = self.powerlist[index] # where powerlist?
        right = self.powerlist[index+1 if cpu % 10 != 0 else index]
        alpha = (cpu/10) - index
        return alpha * right + (1-alpha) * left
    
