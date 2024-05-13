import torch


from MusicGen_Model.LM_Model.LM_Model_T5_CodeBookPattern import LM_Model_T5_CodeBookPattern
from MusicGen_Model.Optimizer.Loss_Function import Loss_Function
from MusicGen_Model.Optimizer.Optimizer import Optimizer
from Config.Config import device, ntoken, d_model, nheads, nlayer, d_hid, dropout, ignore_index, lr, betas, eps, PATH, text_condition_max_length, melody_condition_max_length, word_dropout


transformer = LM_Model_T5_CodeBookPattern(
    word_dropout=word_dropout, tgt_ntoken=ntoken,
    d_model=d_model, nhead=nheads, nlayer=nlayer, d_hid=d_hid, dropout=dropout,
    text_condotion_max_length=text_condition_max_length, melody_condition_max_length=melody_condition_max_length)

criterion = Loss_Function(ignore_index)
optimizer = Optimizer(transformer, lr, betas, eps)

encoder_input = ["This music clip is of a male vocalist doing a beat box. The tempo is medium fast with the vocalist imitating the digital drums, turntable and Dj mixer to a perfection. This vocal percussion is youthful and engaging.",
                 "This music clip is of a male vocalist doing a beat box. The tempo is medium fast with the vocalist imitating the digital drums, turntable and Dj mixer to a perfection. This vocal percussion is rhythmic",
                 "This music clip is of a male vocalist doing a beat box. The tempo is medium fast with the vocalist imitating the digital drums, turntable and Dj mixer to a perfection."]
decoder_input = torch.randint(1, ntoken, (3, 1000))
tgt_data = torch.randint(1, ntoken, (3, 4, 1000))

transformer.train()

for epoch in range(10):
    optimizer.zero_grad()
    output = transformer(
        encoder_input, decoder_input[:, :])

    loss = 0
    for codebook in range(4):
        loss += criterion(output[codebook].contiguous().view(-1, ntoken),
                          tgt_data[:, codebook, :].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

torch.save(transformer.state_dict(), PATH)
