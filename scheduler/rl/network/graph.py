from network import conv_mlp_net, dueling_q_network
import torch
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torchviz import make_dot
import torch.nn as nn

def architect():
    state_size = 3009
    action_size = 10
    model = dueling_q_network(state_size=state_size, action_size=action_size)
    x = torch.randn(1, state_size, 10)
    writer = SummaryWriter("runs/dueling_q_network")

    # Ghi kiến trúc mạng vào TensorBoard
    writer.add_graph(model, x)
    writer.close()

    # In kiến trúc bằng torchsummary
    summary(model, input_size=(state_size, 10))

    # Tạo sơ đồ bằng torchviz
    out = model(x)
    make_dot(out, params=dict(list(model.named_parameters()))).render("dueling_q_network_architecture", format="png")

def graph():
    state_size = 3009
    action_size = 10
    model = dueling_q_network(state_size=state_size, action_size=action_size)
    x = torch.randn(1, state_size, 10)
    writer = SummaryWriter("runs/dueling_q_net_viz")
    writer.add_graph(model, x)
    writer.close()

graph()