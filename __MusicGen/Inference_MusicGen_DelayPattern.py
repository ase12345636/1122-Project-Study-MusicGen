import torch

from MusicGenModel.MusicGen.MusicGen import MusicGen
from MusicGenModel.Compressor.Compressor import Compressor
from Config.Config import delay_pattern_ntoken, d_model, nheads, nlayer, d_hid, dropout, PATH, text_condition_max_length, melody_condition_max_length


mode = "Delay"

transformer = MusicGen(
    tgt_ntoken=delay_pattern_ntoken,
    d_model=d_model,
    nhead=nheads,
    nlayer=nlayer,
    d_hid=d_hid,
    dropout=dropout,
    melody_condition_max_length=melody_condition_max_length+3)
transformer.load_state_dict(torch.load(PATH+"DelayModel.h5"))

with torch.no_grad():
    mem_data = torch.load('Dataset\\PreproccessedData\\train\\MemData\\mem.pt',
                          map_location="cpu", weights_only=True)[0, 0, :]
    mem_data = torch.reshape(mem_data, (1, text_condition_max_length, 512))

    output = transformer.generation(
        mem_data, melody_condition_max_length+4)

    compressor = Compressor(mode=mode)
    compressor.decompress(output, "Dataset\\InferenceData\\DelayOutput.wav")
