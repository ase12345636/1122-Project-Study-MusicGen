import torch


device = torch.device('cuda')

text_condition_max_length = 300
melody_condition_max_length = 500

word_dropout = 0.2

ntoken = 2049
d_model = 512
nheads = 8
nlayer = 10
d_hid = 2048
dropout = 0.2

ignore_index = 2049

lr = 0.0001
betas = (0.9, 0.98)
eps = 1e-9

PATH = "./ModelSave/model.h5"

top_k = 250
temperature = 1.0
