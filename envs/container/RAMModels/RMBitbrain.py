from dc.env.container.RAMModels.RM import RM

class RMBitbrain(RM):
    def __init__(self, size_list, read_list, write_list):
        super().__init__()

        self.size_list = size_list
        self.read_list = read_list
        self.write_list = write_list

    def ram(self):
        pass