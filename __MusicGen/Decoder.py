import torch.nn as nn
import math

from torch import Tensor
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from Absolute_Positional_Encoding import PositionalEncoding


class Decoder(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayer: int, dropout: float = 0.5):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
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

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(
                len(src)).to('cuda')
        output = self.transformer_decoder(src, src_mask)
        output = self.linear(output)
        return output
