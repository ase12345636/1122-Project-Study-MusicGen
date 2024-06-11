import math

from torch import optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from torch import nn
from Config.Config import warm_up_epoch, num_epoch


class CosineLRScheduler(_LRScheduler):
    '''
    Class for handle learning rate
    '''

    def __init__(self, optimizer: Optimizer, total_steps: int, warmup_steps: int,
                 lr_min_ratio: float = 0.0, cycle_length: float = 1.0):

        # Initialize
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps+1
        self.lr_min_ratio = lr_min_ratio
        self.cycle_length = cycle_length

        super().__init__(optimizer)

    def lr_shedualer(self, lr: float, step: int):

        step += 1

        # Warm up
        if step < self.warmup_steps:
            lr_ratio = step/self.warmup_steps
            lr = lr_ratio*lr

        # Cosine learning rate
        elif step <= self.total_steps:
            s = (step-self.warmup_steps) /\
                (self.total_steps - self.warmup_steps)
            lr_ratio = self.lr_min_ratio+0.5*(1 - self.lr_min_ratio) *\
                (1.0+math.cos(math.pi*s/self.cycle_length))
            lr = lr_ratio*lr

        # Stop update learning rate
        else:
            lr = self.lr_min_ratio

        return lr

    def get_lr(self):

        # Compute new learning rate
        return [self.lr_shedualer(lr, self.last_epoch) for lr in self.base_lrs]


def Optimizer(model: nn.Module, lr, betas, eps):

    # Adanw optimizer
    optimizer = optim.AdamW(model.parameters(),
                            lr=lr, betas=betas, eps=eps)

    # Warm up with cosine learning rate schedualer
    scheduler = CosineLRScheduler(
        optimizer, total_steps=num_epoch, warmup_steps=warm_up_epoch)

    return optimizer, scheduler
