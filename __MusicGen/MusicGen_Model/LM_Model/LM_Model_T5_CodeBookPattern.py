import torch
import torch.nn as nn
from torch.distributions import Categorical

from ..Encoder_Model.Encoder_T5 import Encoder_T5
from ..Decoder_Model.Decoder_CodeBookPattern import Decoder_CodeBookPattern


class LM_Model_T5_CodeBookPattern(nn.Module):
    def __init__(self, word_dropout: float, tgt_ntoken: int, d_model: int, nhead: int, nlayer: int, d_hid: int, dropout: int,
                 text_condotion_max_length: int = 100, melody_condition_max_length: int = 1000):
        super(LM_Model_T5_CodeBookPattern, self).__init__()
        self.melody_condition_max_length = melody_condition_max_length
        self.encoder = Encoder_T5(
            d_model=d_model, finetune=True, word_dropout=word_dropout,
            max_length=text_condotion_max_length)
        self.decoder = Decoder_CodeBookPattern(
            ntoken=tgt_ntoken, d_model=d_model, nhead=nhead, d_hid=d_hid, nlayer=nlayer,
            dropout=dropout, max_length=melody_condition_max_length)

    def forward(self, src: list[str], tgt):
        output = self.decoder(tgt, self.encoder(src),
                              nn.Transformer.generate_square_subsequent_mask(self.melody_condition_max_length))
        return output

    def generation(self, src: list[str]):
        with torch.no_grad():
            tgt = torch.tensor(
                [[[2048], [2048], [2048], [2048]]], dtype=torch.int)
            for step in range(1, 50):
                output_logit = self.decoder(
                    tgt, self.encoder(src), nn.Transformer.generate_square_subsequent_mask(step), "generation")
                prediction = torch.tensor([], dtype=torch.int)
                for codebook in range(4):
                    logit, indices = torch.topk(
                        output_logit[:, codebook, :], 20)
                    logit = Categorical(
                        logit / 1.0).sample()
                    prediction = torch.cat((prediction, logit), 0)
                prediction = torch.reshape(prediction, (1, 4, 1))
                tgt = torch.cat((tgt, prediction), 2)

            return torch.reshape(tgt, (1, 1, 4, -1))[:, :, :, 1:]
