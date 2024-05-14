import torch.nn as nn


def Loss_Function(ignore_index: int):
    return nn.CrossEntropyLoss(ignore_index=ignore_index)
