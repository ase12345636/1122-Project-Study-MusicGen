import torch.nn as nn
import math

from torch import Tensor
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from ..Positional_Encoding.Absolute_Positional_Encoding import PositionalEncoding


class Decoder(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayer: int, dropout: float = 0.5, max_length: int = 1000):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_length)
        decoder_layers = TransformerDecoderLayer(
            d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayer)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, tgt: Tensor, src: Tensor) -> Tensor:
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        output = self.transformer_decoder(tgt, src)
        output = self.linear(output)
        return output
