import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.distributions import Categorical # action_space is discrete

class PPOActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim1 = 256, hidden_dim2 = 128):
        super(PPOActorCriticNet, self).__init__()

        # Shared layers
        self.fc1 =  nn.Linear(state_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)

        # Actor
        self.actor_head = nn.Linear(hidden_dim2, action_dim)

        # Critic
        self.critic_head = nn.Linear(hidden_dim2, 1)

    def forward(self, state):
        print("state in net: ",state)
        print("state in net shape: ", state.shape)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_logits = self.actor_head(x)
        value = self.critic_head(x)

        return action_logits, value

    def get_action_distribution(self, state):
        action_logits, _ = self.forward(state)
        return Categorical(logits=action_logits)
    
    def get_value(self, state):
        _, value = self.forward(state)
        return value 