from network.network import dueling_q_network, single_q_network
import torch
from torch import nn
    
class SingleQNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c = input_dim
        self.block_num = 3
        self.online = self.__build_nn(c, output_dim)
        self.target = self.__build_nn(c, output_dim)
        self.target.load_state_dict(self.online.state_dict())

        for p in self.target.parameters():
            p.requires_grad = False
        
    def forward(self, input, model):
        # print(input)
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)


    def __build_nn(self, c, output_dim):
        return single_q_network(input_dims=c, n_actions=output_dim)
    
class DuelingQNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c = input_dim
        self.block_num = 3
        self.online = self.__build_nn(c, output_dim)
        self.target = self.__build_nn(c, output_dim)
        self.target.load_state_dict(self.online.state_dict())

        for p in self.target.parameters():
            p.requires_grad = False
        
    def forward(self, input, model):
        # print(input)
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)


    def __build_nn(self, c, output_dim):
        return dueling_q_network(input_dims=c, n_actions=output_dim)