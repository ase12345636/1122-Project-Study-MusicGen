import torch.nn as nn


from Encoder_T5 import Encoder_T5
from Decoder import Decoder


class LM_Model_T5(nn.Module):
    def __init__(self, word_dropout: float, tgt_ntoken: int, d_model: int, nhead: int, nlayer: int, d_hid: int, dropout: int):
        super(LM_Model_T5, self).__init__()

        self.encoder = Encoder_T5(
            d_model=d_model, finetune=True, word_dropout=word_dropout)
        self.decoder = Decoder(
            ntoken=tgt_ntoken, d_model=d_model, nhead=nhead, d_hid=d_hid, nlayer=nlayer, dropout=dropout)

    def forward(self, src: list[str], tgt):
        # print(type(src))
        output = self.decoder(tgt, self.encoder(src))
        return output
