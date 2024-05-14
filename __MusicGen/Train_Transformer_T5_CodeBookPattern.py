import torch

from MusicGen_Model.Compresion_Model.Compressor import Compressor
from MusicGen_Model.LM_Model.LM_Model_T5_CodeBookPattern import LM_Model_T5_CodeBookPattern
from MusicGen_Model.Optimizer.Loss_Function import Loss_Function
from MusicGen_Model.Optimizer.Optimizer import Optimizer
from Config.Config import device, ntoken, d_model, nheads, nlayer, d_hid, dropout, ignore_index, lr, betas, eps, PATH, text_condition_max_length, melody_condition_max_length, word_dropout


compressor = Compressor()

transformer = LM_Model_T5_CodeBookPattern(
    word_dropout=word_dropout, tgt_ntoken=ntoken,
    d_model=d_model, nhead=nheads, nlayer=nlayer, d_hid=d_hid, dropout=dropout,
    text_condotion_max_length=text_condition_max_length, melody_condition_max_length=melody_condition_max_length)

criterion = Loss_Function(ignore_index)
optimizer = Optimizer(transformer, lr, betas, eps)

encoder_input = [["A male vocalist sings this passionate song. The tempo is slow with emphatic vocals and an acoustic guitar accompaniment. The song is melodic, emotional,sentimental, passionate,pensive, reflective and deep. This song is Alternative Rock."]]
tgt_data = compressor.compress(
    "D:\\Project\\MachineLearning\\__MusicGen\Dataset\\Music_1.wav")

transformer.train()

for epoch in range(10):
    optimizer.zero_grad()
    output = transformer(
        encoder_input, tgt_data[0, :, :, :-1])

    loss = 0
    for codebook in range(4):
        loss += criterion(output[codebook].contiguous().view(-1, ntoken),
                          tgt_data[0, :, codebook, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

torch.save(transformer.state_dict(), PATH)
