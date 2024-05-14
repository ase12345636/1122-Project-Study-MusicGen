import torch.nn as nn
import math

from torch import Tensor
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from ..Positional_Encoding.Absolute_Positional_Encoding import PositionalEncoding


class Decoder_CodeBookPattern(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayer: int, dropout: float = 0.5, max_length: int = 1000):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_length)
        decoder_layers = TransformerDecoderLayer(
            d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayer)
        self.embedding_1 = nn.Embedding(ntoken, d_model)
        self.embedding_2 = nn.Embedding(ntoken, d_model)
        self.embedding_3 = nn.Embedding(ntoken, d_model)
        self.embedding_4 = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear_code_book_1 = nn.Linear(d_model, ntoken)
        self.linear_code_book_2 = nn.Linear(d_model, ntoken)
        self.linear_code_book_3 = nn.Linear(d_model, ntoken)
        self.linear_code_book_4 = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding_1.weight.data.uniform_(-initrange, initrange)
        self.embedding_2.weight.data.uniform_(-initrange, initrange)
        self.embedding_3.weight.data.uniform_(-initrange, initrange)
        self.embedding_4.weight.data.uniform_(-initrange, initrange)
        self.linear_code_book_1.bias.data.zero_()
        self.linear_code_book_1.weight.data.uniform_(-initrange, initrange)
        self.linear_code_book_2.bias.data.zero_()
        self.linear_code_book_2.weight.data.uniform_(-initrange, initrange)
        self.linear_code_book_3.bias.data.zero_()
        self.linear_code_book_3.weight.data.uniform_(-initrange, initrange)
        self.linear_code_book_4.bias.data.zero_()
        self.linear_code_book_4.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = (self.embedding_1(src[:, 0])+self.embedding_2(src[:, 1])
               + self.embedding_3(src[:, 2])+self.embedding_4(src[:, 3]))
        src *= math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(
                len(src)).to('cuda')
        transformer_output = self.transformer_decoder(src, src_mask)
        output_code_book_1 = self.linear_code_book_1(transformer_output)
        output_code_book_2 = self.linear_code_book_2(transformer_output)
        output_code_book_3 = self.linear_code_book_3(transformer_output)
        output_code_book_4 = self.linear_code_book_4(transformer_output)
        return [output_code_book_1, output_code_book_2, output_code_book_3, output_code_book_4]
