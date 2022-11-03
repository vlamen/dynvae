import torch
from torch import nn


class GradNet(nn.Module):
    """
    Class for transforming a neural network or function taking torch Tensors
    and producing the gradient of said network or function
    """
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        with torch.set_grad_enabled(True):
            if not x.requires_grad:
                x.requires_grad_()
            out = self.net(x)
            return torch.autograd.grad(out.sum(), x, create_graph=True)[0]
