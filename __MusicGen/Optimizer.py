import torch.nn as nn
import torch.optim as optim


def Optimizer(model: nn.Module, lr, betas, eps):
    return optim.Adam(model.parameters(),
                      lr=lr, betas=betas, eps=eps)
