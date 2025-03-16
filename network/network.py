from math import ceil
import gymnasium as gym, torch, numpy as np, torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time
class mlp_resblock(nn.Module):
    def __init__(self, in_ch, ch, out_ch=None, block_num=3, is_in=False):
        super().__init__()
        self.models=nn.Sequential()
        self.relus=nn.Sequential()
        self.block_num = block_num
        self.is_in = is_in
        self.is_out = out_ch
        
        if self.is_in:
            self.in_mlp = nn.Sequential(*[
                nn.Linear(in_ch, ch), # layer
                nn.LeakyReLU(0.1, inplace=True)]) # activation function
        for i in range(self.block_num):
            self.models.add_module(str(i), nn.Sequential(*[
                nn.Linear(ch, ch),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(ch, ch)]))
            self.relus.add_module(str(i), nn.Sequential(*[
                nn.LeakyReLU(0.1, inplace=True)]))
        if self.is_out:
            self.out_mlp = nn.Sequential(*[
            nn.Linear(ch, ch), 
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(ch, out_ch)])
            
    def forward(self, x):
        if self.is_in:
            x = self.in_mlp(x)
        for i in range(self.block_num):
            x0 = x
            x = self.models[i](x)
            x += x0
            x = self.relus[i](x)
        if self.is_out:
            x = self.out_mlp(x)
        print(x.shape)
        return x


def main():
    mlp_model = mlp_resblock(in_ch=(512)*67, ch=1024, out_ch=1, block_num=3, is_in=True)
    writer = SummaryWriter("runs/mlp_resblock")

    # Generate a dummy input
    x = torch.randn(1, 34304)

    # Add model to TensorBoard
    writer.add_graph(mlp_model, x)
    writer.close()

if __name__=="__main__":
    main()


class mlp_resblock_relu(nn.Module):
    def __init__(self, in_ch, ch, out_ch=None, block_num=3, is_in=False, is_relu=True):
        super().__init__()
        self.models=nn.Sequential()
        self.relus=nn.Sequential()
        self.block_num = block_num
        self.is_in = is_in
        self.is_out = out_ch
        self.is_relu = is_relu
        
        if self.is_in:
            self.in_mlp = nn.Sequential(*[
                nn.Linear(in_ch, ch), 
                nn.LeakyReLU(0.1, inplace=True)])
        for i in range(self.block_num):
            self.models.add_module(str(i), nn.Sequential(*[
                nn.Linear(ch, ch),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(ch, ch)]))
            self.relus.add_module(str(i), nn.Sequential(*[
                nn.LeakyReLU(0.1, inplace=True)]))
        if self.is_out:
            self.out_mlp = nn.Sequential(*[
            nn.Linear(ch, ch), 
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(ch, out_ch)
            ])
        if self.is_relu:
            self.relu = nn.ReLU(inplace=True)
            
    def forward(self, x):
        if self.is_in:
            x = self.in_mlp(x)
        for i in range(self.block_num):
            x0 = x
            x = self.models[i](x)
            x += x0
            x = self.relus[i](x)
        if self.is_out:
            x = self.out_mlp(x)
        if self.is_relu:
            x = self.relu(x)
        return x



class conv_resblock(nn.Module):
    def __init__(self, in_ch=None, ch=256, out_ch=None, block_num=3, is_relu=True):
        super().__init__()
        self.models=nn.Sequential()
        self.relus=nn.Sequential()
        self.block_num = block_num
        self.is_in = in_ch
        self.is_out = out_ch
        self.is_relu = is_relu
        
        if self.is_in:
            self.in_conv = nn.Sequential(*[
            nn.Conv1d(in_ch, ch, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=False)])
        for i in range(self.block_num):
            self.models.add_module(str(i), nn.Sequential(*[
                nn.Conv1d(ch, ch, kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(0.1, inplace=False),
                nn.Conv1d(ch, ch, kernel_size=1, stride=1, padding=0)]))
            self.relus.add_module(str(i), nn.Sequential(*[
                nn.LeakyReLU(0.1, inplace=False)]))
        if self.is_out:
            self.out_conv = nn.Conv1d(ch, out_ch, kernel_size=1, stride=1, padding=0)
        if self.is_relu:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        if self.is_in:
            x = x.to(dtype=torch.float32)
            x = self.in_conv(x)
        for i in range(self.block_num):
            x0 = x
            x = self.models[i](x)
            x += x0
            x = self.relus[i](x)
        if self.is_out:
            x = self.out_conv(x)
        if self.is_relu:
            x = self.relu(x)
        return x


class conv_mlp_net(nn.Module):
    def __init__(self, conv_in, conv_ch, mlp_in, mlp_ch, out_ch, block_num=3, is_gpu=True):
        super().__init__()

        input_dim = 36
        action_dim = 2
        self.is_gpu = is_gpu
        self.mlp_in = mlp_in
        self.feature_network_A = conv_resblock(in_ch=conv_in, ch=conv_ch, out_ch=conv_ch, block_num=block_num)
        self.feature_network_B = mlp_resblock(in_ch=mlp_in, ch=mlp_ch, out_ch=out_ch, block_num=block_num, is_in=True)

    def load_model(self, filename):
        map_location=lambda storage, loc:storage
        self.load_state_dict(torch.load(filename, map_location=map_location))
        print('load model!')
    
    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
        # print('save model!')

    def forward(self, obs):
        x = self.feature_network_A(obs)
        print(x.shape)
        x = x.view(-1, self.mlp_in)
        print(x.shape, end='\n\n')
        x = self.feature_network_B(x)
        
        return x
   
class dueling_q_network(nn.Module):
    def __init__(self, state_size, action_size, ch = 512):
        super().__init__()
        self.models=nn.Sequential()
        self.relus=nn.Sequential()
        self.output_size = action_size
        
        self.inputLayer = nn.Sequential(*[
            nn.Conv1d(state_size, ch, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=False)])

        for i in range(2):
            self.models.add_module(str(i), nn.Sequential(*[
                nn.Conv1d(ch, ch, kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv1d(ch, ch, kernel_size=1, stride=1, padding=0)]))
            self.relus.add_module(str(i), nn.Sequential(*[
                nn.LeakyReLU(0.1, inplace=True)]))

        self.Value = self.outputAdvantage = nn.Sequential(*[
            nn.Conv1d(ch, ch, kernel_size=1, stride=1, padding=0),            
            nn.LeakyReLU(0.1, inplace=True)]) # activation function

        self.Advantage = self.outputAdvantage = nn.Sequential(*[
            nn.Conv1d(ch, ch, kernel_size=1, stride=1, padding=0),       
            nn.LeakyReLU(0.1, inplace=True)]) # activation function

        self.outputAdvantage = nn.Sequential(*[
            nn.Conv1d(ch, action_size, kernel_size=1, stride=1, padding=0),            
            nn.LeakyReLU(0.1, inplace=True)]) # activation function

    def forward(self, x):
        x = x.float()
        x = self.inputLayer(x)
        for i in range(2):
            x0 = x
            x = self.models[i](x)
            x += x0
            x = self.relus[i](x)
        
        xVal = self.Value(x)
        xVal = self.outputAdvantage(x)
        xAdv = self.Advantage(x)
        xAdv = self.outputAdvantage(x)

        avg = torch.mean(xAdv, dim = 1, keepdim=True)
        q = xAdv + xVal - avg        
        print(q.shape)
        return q

"""
class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed=3152025, fc1_units=256, fc2_units=128, fc3_units=512):
        super(DuelingQNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)

        self.fc_adv = nn.Linear(fc2_units, fc3_units)
        self.fc_val = nn.Linear(fc2_units, fc3_units)

        self.adv = nn.Linear(fc3_units, action_size)
        self.val = nn.Linear(fc3_units, 1)

    def forward(self, state):
        x = torch.transpose(state, 1, 2)
        x = x.float()
        x = nn.LeakyReLU(0.1, inplace=False)(self.fc1(x))
        x = nn.LeakyReLU(0.1, inplace=False)(self.fc2(x))

        x_adv = nn.LeakyReLU(0.1, inplace=False)(self.fc_adv(x))
        x_adv = nn.LeakyReLU(0.1, inplace=False)(self.adv(x_adv))

        x_val = nn.LeakyReLU(0.1, inplace=False)(self.fc_val(x))
        x_val = nn.LeakyReLU(0.1, inplace=False)(self.adv(x_val))

        x_val = torch.transpose(x_val, 1, 2)
        x_adv = torch.transpose(x_adv, 1, 2)

        advAverage = torch.mean(x_adv, dim=1, keepdim=True)
        q = x_adv + x_val - advAverage
        return q[0]
"""