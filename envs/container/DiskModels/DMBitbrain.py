from DM import DM

class DMBitbrain(DM):
    def __init__(self, constant_size, read_list, write_list):
        super().__init__()
        self.constant_size = constant_size
        self.read_list = read_list
        self.write_list = write_list

    def disk(self):
        pass