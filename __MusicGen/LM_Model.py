import torch.nn as nn


from Encoder import Encoder
from Decoder import Decoder


class LM_Model(nn.Module):
    def __init__(self, src_ntoken: int, tgt_ntoken: int, d_model: int, nhead: int, nlayer: int, d_hid: int, dropout: int,
                 text_condotion_max_length: int = 100, melody_condition_max_length: int = 1000):
        super(LM_Model, self).__init__()

        self.encoder = Encoder(
            ntoken=src_ntoken, d_model=d_model, nhead=nhead, d_hid=d_hid, nlayer=nlayer, dropout=dropout, max_length=text_condotion_max_length)
        self.decoder = Decoder(
            ntoken=tgt_ntoken, d_model=d_model, nhead=nhead, d_hid=d_hid, nlayer=nlayer, dropout=dropout, max_length=melody_condition_max_length)

    def forward(self, src, tgt):
        output = self.decoder(tgt, self.encoder(src))
        return output
