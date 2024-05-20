import torch

from MusicGenModel.MusicGen.MusicGen import MusicGen
from MusicGenModel.Compressor.Compressor import Compressor
from Config.Config import mode_table
from Config.Config import validation_dataset_batch_size
from Config.Config import device, delay_pattern_ntoken, d_model, nheads, nlayer, d_hid, dropout, PATH


'''
Change mode to delay pattern mode or parallel pattern mode :

    0 -> delay pattern mode
    1 ->parallel pattern mode
'''
mode_list = mode_table[0]

# Load mode, max length
mode = mode_list[0]
text_condition_max_length = mode_list[1]
melody_condition_max_length = mode_list[2]

# Load model
transformer = MusicGen(
    tgt_ntoken=delay_pattern_ntoken,
    d_model=d_model,
    nhead=nheads,
    nlayer=nlayer,
    d_hid=d_hid,
    dropout=dropout,
    melody_condition_max_length=melody_condition_max_length).to(device)
transformer.load_state_dict(torch.load(PATH+mode+"Model.h5"))

# Start inference
with torch.no_grad():
    for example in range(validation_dataset_batch_size):
        mem_data = torch.load('Dataset\\PreproccessedData\\validation\\MemData\\mem.pt',
                              map_location=device, weights_only=True)[0, example, :]
        mem_data = torch.reshape(mem_data, (1, text_condition_max_length, -1))

        output = transformer.generation(mem_data)

        compressor = Compressor(mode=mode)
        compressor.decompress(
            output, "Dataset\\InferenceData\\validation\\"+mode+"Pattern\\Test"+mode+"Output_"+str(example)+".wav")
