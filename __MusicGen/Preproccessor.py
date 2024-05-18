import json
import torch

from MusicGenModel.EncoderModel.Encoder_T5 import Encoder_T5
from MusicGenModel.Compressor.Compressor import Compressor
from Config.Config import text_condition_max_length, word_dropout, melody_condition_max_length

'''
mem input :
    shape : [batch, batch size, length, hidden dim]
'''

# Prepare mem input source
mem_data = []
for i in range(1, 4):
    file_path = "Dataset\\OriginalData\\Music_"+str(i) + ".wav.json"
    data = json.load(open(file_path))
    mem_data.append(data["caption"])
mem_data = [[mem_data]]

# Convert to mem input
t5 = Encoder_T5(finetune=False,
                word_dropout=word_dropout,
                max_length=text_condition_max_length)
mem_data = t5.forward(mem_data[0])
# print(mem_data.shape)

# Save data
torch.save(mem_data, 'Dataset\\PreproccessedData\\mem.pt')


'''
tgt input :
    shape : [batch, batch size, length, codebook]
'''

# Prepare tgt input source
tgt_data = []
for i in range(1, 4):
    file_path = "Dataset\\OriginalData\\Music_"+str(i) + ".wav"
    tgt_data.append(file_path)
tgt_data = [tgt_data]

# Convert to tgt input
compressor = Compressor(max_length=melody_condition_max_length)
tgt_data = compressor.compress(tgt_data)
# print(tgt_data.shape)

# Save data
torch.save(tgt_data, 'Dataset\\PreproccessedData\\tgt.pt')
