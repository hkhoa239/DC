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
        alpha = (cpu/10) - index
        if cpu > 0:
            static_power = cpu
        else:
            static_power = 0
        if cpu < 70:
            dynamic_power = cpu * alpha
        else:
            dynamic_power = 70 * alpha + ((cpu - 70) ** 2) * (1 - alpha)
        # print("alpha", alpha)
        # print("static", static_power)
        # print("dynamic", dynamic_power)
        return static_power + dynamic_power
    
