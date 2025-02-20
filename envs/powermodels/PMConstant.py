from .PM import PM

class PMConstant(PM):
    def __init__(self, constant):
        super().__init__()
        self.constant = constant
        self.powerlist = [constant] * 11

    def power(self):
        return self.constant