import torch
import torch.nn as nn

from copy import deepcopy


class Model_Ema(nn.Module):
    '''
    Class for update model's parameter with Exponential moving average
    '''

    def __init__(self, model, decay=0.99):
        super(Model_Ema, self).__init__()

        # Initialize
        self.model = deepcopy(model)
        self.model.eval()
        self.decay = decay

    def update_fn(self, parameter_ema, parameter_org):

        # Exponential moving average
        return self.decay * parameter_ema + (1. - self.decay) * parameter_org

    def update(self, model):

        # Update ema model's parameter with exponential moving average
        with torch.no_grad():

            # Trace all model's parameter
            for parameter_ema, parameter_org in zip(self.model.state_dict().values(), model.state_dict().values()):

                # Compute new parameter and cover old one
                parameter_ema.copy_(
                    self.update_fn(parameter_ema, parameter_org))
