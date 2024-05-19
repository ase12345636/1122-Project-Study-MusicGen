import torch

from MusicGenModel.MusicGen.MusicGen import MusicGen
from MusicGenModel.Compressor.Compressor import Compressor
from Config.Config import device, parallel_pattern_ntoken, d_model, nheads, nlayer, d_hid, dropout, PATH, text_condition_max_length, melody_condition_max_length


mode = "Parallel"

transformer = MusicGen(
    tgt_ntoken=parallel_pattern_ntoken,
    d_model=d_model,
    nhead=nheads,
    nlayer=nlayer,
    d_hid=d_hid,
    dropout=dropout,
    melody_condition_max_length=melody_condition_max_length).to(device)
transformer.load_state_dict(torch.load(PATH+"ParallelModel.h5"))

with torch.no_grad():
    mem_data = torch.load('Dataset\\PreproccessedData\\train\\MemData\\mem.pt',
                          map_location=device, weights_only=True)[0, 0, :]
    mem_data = torch.reshape(mem_data, (1, text_condition_max_length, -1))

    output = transformer.generation(mem_data)

    compressor = Compressor(mode=mode)
    compressor.decompress(
        output, "Dataset\\InferenceData\\train\\ParallelPattern\\TestParallelOutput.wav")
