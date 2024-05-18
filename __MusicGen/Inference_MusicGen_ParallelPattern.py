import torch

from MusicGenModel.MusicGen.MusicGen_ParallelPattern import MusicGen_DelayPattern
from MusicGenModel.Compressor.Compressor import Compressor
from Config.Config import device, ntoken, d_model, nheads, nlayer, d_hid, dropout, PATH, text_condition_max_length, melody_condition_max_length


transformer = MusicGen_DelayPattern(
    tgt_ntoken=ntoken,
    d_model=d_model,
    nhead=nheads,
    nlayer=nlayer,
    d_hid=d_hid,
    dropout=dropout,
    melody_condition_max_length=melody_condition_max_length)
transformer.load_state_dict(torch.load(PATH))

with torch.no_grad():
    mem_data = torch.load('Dataset\\PreproccessedData\\mem.pt',
                          map_location="cpu", weights_only=True)[0, 0, :]
    mem_data = torch.reshape(mem_data, (1, text_condition_max_length, 512))

    output = transformer.generation(
        mem_data, melody_condition_max_length+1)

    # output = transformer.generation(
    #     mem_data, 2)

    compressor = Compressor()
    compressor.decompress(output)
