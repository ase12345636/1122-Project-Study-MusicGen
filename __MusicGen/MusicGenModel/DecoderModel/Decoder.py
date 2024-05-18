import torch
import torch.nn as nn
import math

from torch import Tensor
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from ..PositionalEncoding.Absolute_Positional_Encoding import PositionalEncoding


class Decoder(nn.Module):
    '''
    Class for handle text condition
    '''

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayer: int, dropout: float = 0.5, max_length: int = 500):
        # Initialize
        super().__init__()
        self.d_model = d_model
        self.ntoken = ntoken
        self.max_length = max_length

        self.embedding_1 = nn.Embedding(self.ntoken, d_model)
        self.embedding_2 = nn.Embedding(self.ntoken, d_model)
        self.embedding_3 = nn.Embedding(self.ntoken, d_model)
        self.embedding_4 = nn.Embedding(self.ntoken, d_model)

        self.pos_encoder = PositionalEncoding(d_model, dropout, max_length)

        self.linear = nn.Linear(512, d_model)

        decoder_layers = TransformerDecoderLayer(
            d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayer)

        self.linear_decoder = nn.Linear(d_model, d_model)

        self.linear_code_book_1 = nn.Linear(d_model, self.ntoken-1)
        self.linear_code_book_2 = nn.Linear(d_model, self.ntoken-1)
        self.linear_code_book_3 = nn.Linear(d_model, self.ntoken-1)
        self.linear_code_book_4 = nn.Linear(d_model, self.ntoken-1)

        self.init_weights()

    def init_weights(self) -> None:
        # Initialize
        initrange = 0.1
        self.embedding_1.weight.data.uniform_(-initrange, initrange)
        self.embedding_2.weight.data.uniform_(-initrange, initrange)
        self.embedding_3.weight.data.uniform_(-initrange, initrange)
        self.embedding_4.weight.data.uniform_(-initrange, initrange)

        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

        self.linear_decoder.bias.data.zero_()
        self.linear_decoder.weight.data.uniform_(-initrange, initrange)

        self.linear_code_book_1.bias.data.zero_()
        self.linear_code_book_1.weight.data.uniform_(-initrange, initrange)
        self.linear_code_book_2.bias.data.zero_()
        self.linear_code_book_2.weight.data.uniform_(-initrange, initrange)
        self.linear_code_book_3.bias.data.zero_()
        self.linear_code_book_3.weight.data.uniform_(-initrange, initrange)
        self.linear_code_book_4.bias.data.zero_()
        self.linear_code_book_4.weight.data.uniform_(-initrange, initrange)

    def forward(self,  tgt: Tensor, mem: Tensor, status: str = "train") -> Tensor:
        '''
        Input : tgt
            A batch of Decoder input

            shape : [batch size, code book, length]

                mem
            A batch of Encoder output

            shape : [batch size, length, hidden size]

    ->  Processed tgt : processed_tgt
            A batch of processed Decoder input

            shape : [batch size, length, hidden size]

    ->  Decoder output : decoder_output
            A batch of latent codes from Decoder

            shape : [batch size, length, hidden size]

    ->  Final Prediction : prediction_code_book_1, 2, 3, 4
            A batch of logits prediction of 4 codebooks

            shape : [batch size, code book, length, hidden size]
        '''

        # Get mask for masked attention
        mask = nn.Transformer.generate_square_subsequent_mask(
            len(tgt[0, 0, :]))

        # Compute embeddings of tgt input of each codebooks
        # Compute sum of four codebooks embeddings
        processed_tgt = (self.embedding_1(tgt[:, 0]) * math.sqrt(self.d_model) +
                         self.embedding_2(tgt[:, 1]) * math.sqrt(self.d_model) +
                         self.embedding_3(tgt[:, 2]) * math.sqrt(self.d_model) +
                         self.embedding_4(tgt[:, 3]) * math.sqrt(self.d_model))

        # Compute posision encoding of tgt input
        processed_tgt = self.pos_encoder(processed_tgt)

        # Convert src input from 512 to d_model
        mem = self.linear(mem)

        # Compute decoder's output
        decoder_output = nn.functional.relu(self.linear_decoder(
            self.transformer_decoder(processed_tgt, mem, mask)))

        if status == "train":
            # Compute probbaility distrobute of each codebooks
            output_code_book_1 = torch.reshape(self.linear_code_book_1(
                decoder_output), (-1, 1, self.max_length, self.ntoken-1))
            output_code_book_2 = torch.reshape(self.linear_code_book_2(
                decoder_output), (-1, 1, self.max_length, self.ntoken-1))
            output_code_book_3 = torch.reshape(self.linear_code_book_3(
                decoder_output), (-1, 1, self.max_length, self.ntoken-1))
            output_code_book_4 = torch.reshape(self.linear_code_book_4(
                decoder_output), (-1, 1, self.max_length, self.ntoken-1))

            # Concat each distrobute of codebooks and return
            return torch.cat((output_code_book_1, output_code_book_2, output_code_book_3, output_code_book_4), 1)

        elif status == "generation":
            # Compute probbaility distrobute of each codebooks
            # Take last tokenprobbaility distrobute
            prediction_code_book_1 = torch.reshape(nn.functional.softmax(self.linear_code_book_1(
                decoder_output)[:, -1, :], 1), (1, 1, 1, self.ntoken-1))
            prediction_code_book_2 = torch.reshape(nn.functional.softmax(self.linear_code_book_2(
                decoder_output)[:, -1, :], 1), (1, 1, 1, self.ntoken-1))
            prediction_code_book_3 = torch.reshape(nn.functional.softmax(self.linear_code_book_3(
                decoder_output)[:, -1, :], 1), (1, 1, 1, self.ntoken-1))
            prediction_code_book_4 = torch.reshape(nn.functional.softmax(self.linear_code_book_4(
                decoder_output)[:, -1, :], 1), (1, 1, 1, self.ntoken-1))

            # Concat each distrobute of codebooks and return
            return torch.cat((prediction_code_book_1, prediction_code_book_2, prediction_code_book_3, prediction_code_book_4), 1)
