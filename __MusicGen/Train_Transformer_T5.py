import torch


from LM_Model_T5 import LM_Model_T5
from Loss_Function import Loss_Function
from Optimizer import Optimizer
from Config import device, ntoken, d_model, nheads, nlayer, d_hid, dropout, ignore_index, lr, betas, eps, PATH


transformer = LM_Model_T5(
    word_dropout=0.2, tgt_ntoken=ntoken, d_model=d_model, nhead=nheads, nlayer=nlayer, d_hid=d_hid, dropout=dropout)

criterion = Loss_Function(ignore_index)
optimizer = Optimizer(transformer, lr, betas, eps)

src_data = ["This music clip is of a male vocalist doing a beat box. The tempo is medium fast with the vocalist imitating the digital drums, turntable and Dj mixer to a perfection. This vocal percussion is lively, rhythmic, youthful and engaging."]
tgt_data = torch.randint(1, ntoken, (1, 100))


transformer.train()

for epoch in range(10):
    optimizer.zero_grad()
    # print(type(src_data))
    output = transformer(src_data, tgt_data)
    loss = criterion(output.contiguous().view(-1, ntoken),
                     tgt_data.contiguous().view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

torch.save(transformer.state_dict(), PATH)
