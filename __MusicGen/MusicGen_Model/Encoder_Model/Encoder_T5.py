import torch
import torch.nn as nn
import typing as tp
import random

from transformers import AutoTokenizer, T5EncoderModel

# a text condition can be a string or None (if doesn't exist)
TextCondition = tp.Optional[str]
ConditionType = tp.Tuple[torch.Tensor, torch.Tensor]  # condition, mask


class Encoder_T5(nn.Module):
    def __init__(self,  d_model: int, finetune: bool,
                 word_dropout: float = 0., max_length: int = 100):
        super().__init__()
        self.finetune = finetune

        self.word_dropout = word_dropout
        self.max_length = max_length

        self.t5_tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        self.t5_encoder = T5EncoderModel.from_pretrained(
            "google-t5/t5-small").train(mode=False)  # .to(torch.device("cuda"))

        self.linear = nn.Linear(512, d_model)

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def tokenize(self, text_condition: list[str]):
        # text_condition_ids = self.t5_tokenizer(
        #     text_condition, return_tensors="pt"
        # ).input_ids

        # new_text_condition_ids = []
        # for example in text_condition_ids:
        #     temp_text_condition_ids = []
        #     for token_id in example:
        #         if random.random() >= 0.0:
        #             temp_text_condition_ids.append(token_id)
        #     for index in range(self.max_length-len(temp_text_condition_ids)):
        #         temp_text_condition_ids.append(0)
        #     new_text_condition_ids.append(temp_text_condition_ids)
        # text_condition_ids = torch.IntTensor(new_text_condition_ids)
        # return text_condition_ids

        text_condition_ids = []
        for example in text_condition[0]:
            temp_text_condition_ids = self.t5_tokenizer(
                example, return_tensors="pt"
            ).input_ids

            new_text_condition_ids = []
            for token_id in temp_text_condition_ids[0]:
                if random.random() >= 0.0:
                    new_text_condition_ids.append(token_id)
            for index in range(self.max_length-len(new_text_condition_ids)):
                new_text_condition_ids.append(0)
            text_condition_ids.append(new_text_condition_ids)
        text_condition_ids = torch.IntTensor(text_condition_ids)
        return text_condition_ids

    def forward(self, text_condition: list[str]) -> ConditionType:
        text_condition_ids = self.tokenize(text_condition)
        output = self.t5_encoder(text_condition_ids)
        output = self.linear(output.last_hidden_state)
        return output
