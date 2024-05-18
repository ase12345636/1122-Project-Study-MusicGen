import torch
import torch.nn as nn
import typing as tp
import random

from transformers import AutoTokenizer, T5EncoderModel

# a text condition can be a string or None (if doesn't exist)
TextCondition = tp.Optional[str]
ConditionType = tp.Tuple[torch.Tensor, torch.Tensor]  # condition, mask


class Encoder_T5(nn.Module):
    '''
    Class for handle text condition
    '''

    def __init__(self, finetune: bool,
                 word_dropout: float = 0., max_length: int = 100):
        # Initialize
        super().__init__()
        self.finetune = finetune

        self.word_dropout = word_dropout
        self.max_length = max_length

        self.t5_tokenizer = AutoTokenizer.from_pretrained(
            "google-t5/t5-small")
        self.t5_encoder = T5EncoderModel.from_pretrained(
            "google-t5/t5-small").train(mode=False)

    def forward(self, mem_input: list[str]):
        '''
        Input : text_condition
            A batch of string

            shape : [batch size]


    ->  Token : text_condition_ids
            A batch of token id

            shape : [batch size, length]

    ->  Output : mem_input
            A batch of latent codes from T5 Encoder

            shape : [batch, batch size, length, hidden size]
        '''

        # Prepare token ids
        text_condition_ids = []

        # Proccess each example
        for example in mem_input[0]:

            # Convert to token ids
            temp_text_condition_ids = self.t5_tokenizer(
                example, return_tensors="pt"
            ).input_ids

            # Randomly drop token ids with word_dropout
            new_text_condition_ids = []
            for token_id in temp_text_condition_ids[0]:
                if (len(new_text_condition_ids) >= self.max_length):
                    break

                if random.random() >= 0.0:
                    new_text_condition_ids.append(token_id)

            # Append <paddin_token_id> to fit max length
            for index in range(self.max_length-len(new_text_condition_ids)):
                new_text_condition_ids.append(0)

            text_condition_ids.append(new_text_condition_ids)

        # Convert to int_tensor due to index must be integer
        text_condition_ids = torch.IntTensor(text_condition_ids)

        # Get word embedding
        mem_input = self.t5_encoder(text_condition_ids)
        mem_input = mem_input.last_hidden_state

        # Permute tgt input and return
        return torch.reshape(mem_input, (1, -1, self.max_length, 512))
