import torch.nn as nn


from ..Encoder_Model.Encoder_T5 import Encoder_T5
from ..Decoder_Model.Decoder import Decoder


class LM_Model_T5(nn.Module):
    def __init__(self, word_dropout: float, tgt_ntoken: int, d_model: int, nhead: int, nlayer: int, d_hid: int, dropout: int,
                 text_condotion_max_length: int = 100, melody_condition_max_length: int = 1000):
        super(LM_Model_T5, self).__init__()

        self.encoder = Encoder_T5(
            d_model=d_model, finetune=True, word_dropout=word_dropout,
            max_length=text_condotion_max_length)
        self.decoder = Decoder(
            ntoken=tgt_ntoken, d_model=d_model, nhead=nhead, d_hid=d_hid, nlayer=nlayer,
            dropout=dropout, max_length=melody_condition_max_length)

    def forward(self, src: list[str], tgt):
        output = self.decoder(tgt, self.encoder(src))
        return output
