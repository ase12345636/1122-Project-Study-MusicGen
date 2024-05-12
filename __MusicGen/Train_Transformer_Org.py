import torch


from LM_Model import LM_Model
from Loss_Function import Loss_Function
from Optimizer import Optimizer
from Config import device, ntoken, d_model, nheads, nlayer, d_hid, dropout, ignore_index, lr, betas, eps, PATH


transformer = LM_Model(
    ntoken, ntoken, d_model, nheads, nlayer, d_hid, dropout).to(device)

criterion = Loss_Function(ignore_index)
optimizer = Optimizer(transformer, lr, betas, eps)


src_data = torch.randint(1, ntoken, (64, 100)).to(device)
tgt_data = torch.randint(1, ntoken, (64, 100)).to(device)


transformer.train()

for epoch in range(5):
    optimizer.zero_grad()
    output = transformer(src_data, tgt_data)
    loss = criterion(output.contiguous().view(-1, ntoken),
                     tgt_data.contiguous().view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

torch.save(transformer.state_dict(), PATH)
