import torch
import torch.nn as nn
import math

from torch import Tensor
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from ..Positional_Encoding.Absolute_Positional_Encoding import PositionalEncoding


class Decoder_CodeBookPattern(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayer: int, dropout: float = 0.5, max_length: int = 1000):
        super().__init__()
        self.ntoken = ntoken
        self.max_length = max_length
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_length)
        decoder_layers = TransformerDecoderLayer(
            d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayer)
        self.embedding_1 = nn.Embedding(self.ntoken, d_model)
        self.embedding_2 = nn.Embedding(self.ntoken, d_model)
        self.embedding_3 = nn.Embedding(self.ntoken, d_model)
        self.embedding_4 = nn.Embedding(self.ntoken, d_model)
        self.d_model = d_model
        self.linear_code_book_1 = nn.Linear(d_model, self.ntoken-1)
        self.linear_code_book_2 = nn.Linear(d_model, self.ntoken-1)
        self.linear_code_book_3 = nn.Linear(d_model, self.ntoken-1)
        self.linear_code_book_4 = nn.Linear(d_model, self.ntoken-1)
        self.logits_predict_layer = nn.Sigmoid()

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

    def forward(self,  tgt: Tensor, src: Tensor, mask, status: str = "train") -> Tensor:
        tgt = (self.embedding_1(tgt[:, 0]) +
               self.embedding_2(tgt[:, 1]) +
               self.embedding_3(tgt[:, 2]) +
               self.embedding_4(tgt[:, 3]))
        tgt *= math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        transformer_output = self.transformer_decoder(tgt, src, mask)
        if status == "train":
            output_code_book_1 = torch.reshape(self.logits_predict_layer(self.linear_code_book_1(
                transformer_output)), (-1, 1, self.max_length, self.ntoken-1))
            output_code_book_2 = torch.reshape(self.logits_predict_layer(self.linear_code_book_2(
                transformer_output)), (-1, 1, self.max_length, self.ntoken-1))
            output_code_book_3 = torch.reshape(self.logits_predict_layer(self.linear_code_book_3(
                transformer_output)), (-1, 1, self.max_length, self.ntoken-1))
            output_code_book_4 = torch.reshape(self.logits_predict_layer(self.linear_code_book_4(
                transformer_output)), (-1, 1, self.max_length, self.ntoken-1))
            return torch.cat((output_code_book_1, output_code_book_2, output_code_book_3, output_code_book_4), 1)

        elif status == "generation":
            prediction_code_book_1 = self.logits_predict_layer(self.linear_code_book_1(
                transformer_output))[:, -1:]
            prediction_code_book_2 = self.logits_predict_layer(self.linear_code_book_2(
                transformer_output))[:, -1:]
            prediction_code_book_3 = self.logits_predict_layer(self.linear_code_book_3(
                transformer_output))[:, -1:]
            prediction_code_book_4 = self.logits_predict_layer(self.linear_code_book_4(
                transformer_output))[:, -1:]
            return torch.cat((prediction_code_book_1, prediction_code_book_2, prediction_code_book_3, prediction_code_book_4), 1)
