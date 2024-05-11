import torch.nn as nn


from Encoder import Encoder
from Decoder import Decoder


class LM_Model(nn.Module):
    def __init__(self, src_ntoken: int, tgt_ntoken: int, d_model: int, nhead: int, nlayer: int, d_hid: int, dropout: int):
        super(LM_Model, self).__init__()

        self.encoder = Encoder(
            ntoken=src_ntoken, d_model=d_model, nhead=nhead, d_hid=d_hid, nlayer=nlayer, dropout=dropout)
        self.decoder = Decoder(
            ntoken=tgt_ntoken, d_model=d_model, nhead=nhead, d_hid=d_hid, nlayer=nlayer, dropout=dropout)

    def forward(self, src, tgt):
        output = self.decoder(tgt, self.encoder(src))
        return output
