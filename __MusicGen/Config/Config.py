import torch


device = torch.device('cuda')


text_condition_max_length = 250
melody_condition_max_length = 250
loss_function_max_length = 250

mode_table = [["Delay", text_condition_max_length, melody_condition_max_length+3],
              ["Parallel", text_condition_max_length, melody_condition_max_length]]


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

parallel_pattern_ntoken = 2049
delay_pattern_ntoken = 2050

output_ntoken = 2048

SOS_token = 2048
SP_token = 2049


d_model = 512
nheads = 8
nlayer = 12
d_hid = 2048
dropout = 0.2


batch_multiplier = 1
warm_up_epoch = 5
num_epoch = 300

lr = 0.0001
betas = (0.9, 0.95)
eps = 1e-9

guidance_scale = 0.0

PATH = ".//ModelSave//"


top_k = 250
temperature = 1.0
