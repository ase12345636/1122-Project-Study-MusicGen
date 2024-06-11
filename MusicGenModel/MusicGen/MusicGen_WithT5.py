import torch
import torch.nn as nn

from transformers import T5EncoderModel
from torch.distributions import Categorical
from torch.nn import TransformerEncoderLayer
from ..DecoderModel.Decoder import Decoder
from Config.Config import device, top_k, temperature, SOS_token, SP_token, guidance_scale


class MusicGen_WithT5(nn.Module):
    '''
    Class for generation audio
    '''

    def __init__(self, tgt_ntoken: int, d_model: int, nhead: int, nlayer: int, d_hid: int, dropout: int,
                 melody_condition_max_length: int = 500):
        super(MusicGen_WithT5, self).__init__()

        # Initialize
        self.melody_condition_max_length = melody_condition_max_length

        self.t5_encoder = T5EncoderModel.from_pretrained(
            "google-t5/t5-large").train(mode=True)

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

        mem = self.t5_encoder(mem).last_hidden_state
        prediction = self.decoder(tgt, mem)

        # print(prediction.shape)

        return prediction

    def generation(self, mem, mode="Delay"):
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
            [[[SOS_token], [SOS_token], [SOS_token], [SOS_token]]]).to(device)

        mem_null = torch.zeros(mem.shape).to(device)

        # Autoregressively generate next token
        for step in range(1, self.melody_condition_max_length+1):

            # Classifier-free guidance
            output_logit_text = self.decoder(
                tgt, mem, "generation")*(1.0-guidance_scale)
            output_logit_null = self.decoder(
                tgt, mem_null, "generation")*guidance_scale

            output_logit = output_logit_text+output_logit_null

            # Top k sample
            # k = 250, temperature = 1.0
            prediction = torch.IntTensor([]).to(device)
            for codebook in range(4):
                logit, indices = torch.topk(
                    output_logit[:, codebook, -1, :], top_k)
                logit = Categorical(
                    logit / temperature).sample()

                if mode == "Delay" and ((tgt.size(2) >= 1 and tgt.size(2) < 1+codebook) or
                                        (tgt.size(2) > self.melody_condition_max_length-(3-codebook))):
                    indices = torch.IntTensor([[SP_token]]).to(device)

                else:
                    indices = torch.reshape(indices[:, logit], (1, 1))

                prediction = torch.cat((prediction, indices), 1)

            # Put new token behind the tgt
            prediction = torch.reshape(prediction, (1, 4, 1))
            tgt = torch.cat((tgt, prediction), 2)

        # Take generated sequence
        # Reshape and return
        return torch.reshape(tgt, (1, 1, 4, -1))[:, :, :, 1:]
