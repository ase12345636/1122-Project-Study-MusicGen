import torch


device = torch.device('cuda')


text_condition_max_length = 250
melody_condition_max_length = 250
loss_function_max_length = 250

delay_pattern_ntoken = 2050
parallel_pattern_ntoken = 2049

mode_table = [["Delay", text_condition_max_length, melody_condition_max_length+3, delay_pattern_ntoken],
              ["Parallel", text_condition_max_length, melody_condition_max_length, parallel_pattern_ntoken]]


training_max_num_batch = 1
training_max_num_batch_size = 30

validation_max_num_batch = 1
validation_max_num_batch_size = 30

test_max_num_batch = 1
test_max_num_batch_size = 30

dir = [["train", training_max_num_batch, training_max_num_batch_size],
       ["validation", validation_max_num_batch, validation_max_num_batch_size],
       ["test", test_max_num_batch, test_max_num_batch_size]]

word_dropout = 0.2

output_ntoken = 2048

SOS_token = 2048
SP_token = 2049

d_model = 4096
nheads = 8
nlayer = 8
d_hid = 2048
dropout = 0.2

batch_multiplier = 11
warm_up_epoch = 10
num_epoch = 300

lr = 5e-4
betas = (0.9, 0.95)
eps = 1e-9

guidance_scale = 0.2

PATH = ".//ModelSave//"

top_k = 250
temperature = 1.0
