import torch


device = torch.device('cuda')

text_condition_max_length = 100
melody_condition_max_length = 1000

word_dropout = 0.2

ntoken = 2048
d_model = 512
nheads = 8
nlayer = 6
d_hid = 2048
dropout = 0.1

ignore_index = 0

lr = 0.0001
betas = (0.9, 0.98)
eps = 1e-9

PATH = "./Model_Save/model.h5"
