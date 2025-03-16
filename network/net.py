from network.network import dueling_q_network, conv_mlp_net
import torch
from torch import nn

class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # print(input_dim)
        c, h = input_dim
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
#        return conv_mlp_net(conv_in=c, conv_ch=512, mlp_in=output_dim*512, mlp_ch=1024, out_ch=output_dim,block_num=self.block_num)
        return dueling_q_network(state_size=c, action_size=output_dim)