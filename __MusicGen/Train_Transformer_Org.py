import torch


from MusicGen_Model.LM_Model.LM_Model import LM_Model
from MusicGen_Model.Optimizer.Loss_Function import Loss_Function
from MusicGen_Model.Optimizer.Optimizer import Optimizer
from Config.Config import device, ntoken, d_model, nheads, nlayer, d_hid, dropout, ignore_index, lr, betas, eps, PATH, text_condition_max_length, melody_condition_max_length


transformer = LM_Model(
    ntoken, ntoken, d_model, nheads, nlayer, d_hid, dropout, text_condition_max_length, melody_condition_max_length).to(device)

criterion = Loss_Function(ignore_index)
optimizer = Optimizer(transformer, lr, betas, eps)


src_data = torch.randint(1, ntoken, (64, 100)).to(device)
tgt_data = torch.randint(1, ntoken, (64, 100)).to(device)


transformer.train()

for epoch in range(5):
    optimizer.zero_grad()
    output = transformer(src_data, tgt_data[:, :-1])
    loss = criterion(output.contiguous().view(-1, ntoken),
                     tgt_data[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

torch.save(transformer.state_dict(), PATH)
