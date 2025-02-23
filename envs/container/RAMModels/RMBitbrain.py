from .RM import RM

class RMBitbrain(RM):
    def __init__(self, size_list, read_list, write_list):
        super().__init__()

        self.size_list = size_list
        self.read_list = read_list
        self.write_list = write_list

    def ram(self): # track ram size, read, write at current time
        size_list_count = (self.container.env.interval - self.container.startAt) % len(self.size_list)
        read_list_count = (self.container.env.interval - self.container.startAt) % len(self.read_list)
        write_list_count = (self.container.env.interval - self.container.startAt) % len(self.write_list)
        return self.size_list[size_list_count], self.read_list[read_list_count], self.write_list[write_list_count]