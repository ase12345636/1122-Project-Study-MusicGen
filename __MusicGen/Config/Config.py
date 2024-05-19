import torch


device = torch.device('cuda')

text_condition_max_length = 250
melody_condition_max_length = 250

word_dropout = 0.2

parallel_pattern_ntoken = 2049
delay_pattern_ntoken = 2050
output_ntoken = 2048

SOS_token = 2048
SP_token = 2049

d_model = 512
nheads = 8
nlayer = 10
d_hid = 2048
dropout = 0.2

lr = 0.0001
betas = (0.9, 0.98)
eps = 1e-9

PATH = ".//ModelSave//"

top_k = 250
temperature = 1.0
