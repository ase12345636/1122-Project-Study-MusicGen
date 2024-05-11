import torch
import torch.nn as nn
import torch.optim as optim

from LM_Model import LM_Model
from Loss_Function import Loss_Function
from Config import ntoken, d_model, nheads, nlayer, d_hid, dropout, ignore_index, PATH


device = torch.device('cuda')

transformer = LM_Model(
    ntoken, d_model, nheads, nlayer, d_hid, dropout).to(device)

transformer.load_state_dict(torch.load(PATH))

criterion = Loss_Function(ignore_index)

transformer.eval()

# Generate random sample validation data
# (seq_length, batch_size)
val_src_data = torch.randint(1, ntoken, (64, 100)).to(device)
val_tgt_data = torch.randint(1, ntoken, (64, 100)).to(device)

with torch.no_grad():

    val_output = transformer(val_src_data, val_tgt_data)
    val_loss = criterion(val_output.contiguous(
    ).view(-1, ntoken), val_tgt_data.contiguous().view(-1))
    print(f"Validation Loss: {val_loss.item()}")
