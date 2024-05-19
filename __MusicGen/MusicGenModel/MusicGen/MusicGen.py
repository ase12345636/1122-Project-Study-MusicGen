import torch
import torch.nn as nn

from torch.distributions import Categorical
from ..DecoderModel.Decoder import Decoder
from Config.Config import top_k, temperature, SOS_token


class MusicGen(nn.Module):
    '''
    Class for generation audio with delay pattern
    '''

    def __init__(self, tgt_ntoken: int, d_model: int, nhead: int, nlayer: int, d_hid: int, dropout: int,
                 melody_condition_max_length: int = 500):
        super(MusicGen, self).__init__()
        # Initialize
        self.melody_condition_max_length = melody_condition_max_length

        self.decoder = Decoder(
            ntoken=tgt_ntoken, d_model=d_model, nhead=nhead, d_hid=d_hid, nlayer=nlayer,
            dropout=dropout, max_length=melody_condition_max_length)

    def forward(self, mem, tgt):
        '''
        input : tgt
            A batch of Decoder input

            shape : [batch size, code book, length]

                mem
            A batch of Encoder output

            shape : [batch size, length, hidden size]

    ->  Einal Prediction : prediction_code_book_1, 2, 3, 4
            A batch of logits prediction of 4 codebooks

            shape : [batch size, code book, length, hidden size]
        '''

        prediction = self.decoder(tgt, mem)
        return prediction

    def generation(self, src, length: int = 500):
        '''
        Input : tgt
            A batch of Decoder input

            shape : [1, code book, length]

                mem
            A batch of Encoder output

            shape : [1, length, hidden size]

    ->  Generation : prediction_code_book_1, 2, 3, 4
            A batch of logits prediction of 4 codebooks

            shape : [1, code book, length, hidden size]
        '''

        # Add SOS token
        tgt = torch.IntTensor(
            [[[SOS_token], [SOS_token], [SOS_token], [SOS_token]]])

        # Autoregressively generate next token
        for step in range(1, length):
            output_logit = self.decoder(
                tgt, src, "generation")

            # Top k sample
            # k = 250, temperature = 1.0
            prediction = torch.IntTensor([])
            for codebook in range(4):
                logit, indices = torch.topk(
                    output_logit[:, codebook, -1, :], top_k)
                logit = Categorical(
                    logit / temperature).sample()
                indices = torch.reshape(indices[:, logit], (1, 1))
                prediction = torch.cat((prediction, indices), 1)

            # Put new token behind the tgt
            prediction = torch.reshape(prediction, (1, 4, 1))
            tgt = torch.cat((tgt, prediction), 2)

        # Take generated sequence
        # Reshape and return
        return torch.reshape(tgt, (1, 1, 4, -1))[:, :, :, 1:]
