import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
   

class single_q_network(nn.Module):
    def __init__(self, n_actions, input_dims):
        super(single_q_network, self).__init__()

        # 3 convolutional layers
        self.conv1 = nn.Conv1d(input_dims[0], 32, kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=1, stride=1)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, stride=1)

        self.conv_output_dims = self.get_conv_output_dimensions(input_dims)

        # Fully connected layers
        self.fc1 = nn.Linear(self.conv_output_dims, 1024)
        self.fc2 = nn.Linear(1024, 512)

        # Output: trực tiếp Q-values cho tất cả actions
        self.q_values = nn.Linear(512, n_actions)

    def get_conv_output_dimensions(self, input_dims):
        temp = torch.zeros(1, *input_dims)
        dim1 = self.conv1(temp)
        dim2 = self.conv2(dim1)
        dim3 = self.conv3(dim2)
        return int(np.prod(dim3.size()))

    def forward(self, data):
        if data.dim() == 2:
            data = data.unsqueeze(0)
        data = data.float()

        x = F.relu(self.conv1(data))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        q = self.q_values(x)
        return q

class dueling_q_network(nn.Module):
    def __init__(self, n_actions, input_dims):
        super(dueling_q_network, self).__init__()

        # 3 convolutional layers
        self.conv1 = nn.Conv1d(input_dims[0], 32, kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=1, stride=1)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, stride=1)

        self.conv_output_dims = self.get_conv_output_dimensions(input_dims)

        # 2 fully-connected layers
        self.fc1 = nn.Linear(self.conv_output_dims, 1024)
        self.fc2 = nn.Linear(1024, 512)

        self.Value = nn.Linear(512, 1)
        self.Advantage = nn.Linear(512, n_actions)

    def get_conv_output_dimensions(self, input_dims):
        temp = torch.zeros(1, *input_dims)
        dim1 = self.conv1(temp)
        dim2 = self.conv2(dim1)
        dim3 = self.conv3(dim2)
        return int(np.prod(dim3.size()))

    def forward(self, data):
        if data.dim() == 2:
            data = data.unsqueeze(0)
        data = data.float()
        conv_layer1 = F.relu(self.conv1(data))
        conv_layer2 = F.relu(self.conv2(conv_layer1))
        conv_layer3 = F.relu(self.conv3(conv_layer2))
        
        output_conv_layer = conv_layer3.view(conv_layer3.size()[0], -1)

        fc_layer1 = F.relu(self.fc1(output_conv_layer))
        fc_layer2 = F.relu(self.fc2(fc_layer1))

        value = self.Value(fc_layer2)
        advantage = self.Advantage(fc_layer2)
        return value, advantage