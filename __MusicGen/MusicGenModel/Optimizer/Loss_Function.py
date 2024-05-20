import torch
import torch.nn as nn

from Config.Config import output_ntoken


def Loss_Function(prediction: torch.tensor, tgt_gt: torch.tensor, max_length: int = 250, mode: str = "Delay"):

    criterion = nn.CrossEntropyLoss()

    # Compute loss with 4 codebooks
    loss = 0.0

    for example in range(len(prediction)):

        for codebook in range(4):

            # Compute loss with parallel mode
            if mode == "Parallel":
                loss += criterion(prediction[example, codebook, :, :].contiguous().view(-1, output_ntoken),
                                  tgt_gt[example, codebook, :].contiguous().view(-1))

            # Compute loss with delay mode
            if mode == "Delay":

                # Skip SP Token
                temp_prediction = prediction[example, codebook, codebook:, :]
                temp_tgt_gt = tgt_gt[example, codebook, codebook:]
                loss += criterion(temp_prediction[:max_length, :].contiguous().view(-1, output_ntoken),
                                  temp_tgt_gt[:max_length].contiguous().view(-1))

    # Update parameters
    loss /= float(len(prediction))

    return loss
