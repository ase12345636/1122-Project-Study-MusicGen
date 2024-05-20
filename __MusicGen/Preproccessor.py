import os

import json
import torch

from MusicGenModel.EncoderModel.Encoder_T5 import Encoder_T5
from MusicGenModel.Compressor.Compressor import Compressor
from Config.Config import dir
from Config.Config import word_dropout, text_condition_max_length, melody_condition_max_length

'''
mem input :
    shape : [batch, batch size, length, hidden dim]
'''

# Prepare mem input source
t5 = Encoder_T5(finetune=False,
                word_dropout=word_dropout,
                max_length=text_condition_max_length)

for dataset in dir:
    text_folder_path = "Dataset\\OriginalData\\json\\"+dataset[0]+"\\"
    text_dirs = os.listdir(text_folder_path)
    text_dirs.sort()

    # For each batch
    mem_data = None
    num = 0
    for batch in range(dataset[1]):

        # For each example
        text_path = []
        for example in range(dataset[2]):
            json_file_path = text_folder_path+text_dirs[num]
            json_data = json.load(open(json_file_path))
            text_path.append(json_data["caption"])
            num += 1

        # Convert to mem input
        text_path = [[text_path]]
        text_data = t5(text_path[0])

        if mem_data == None:
            mem_data = text_data

        else:
            mem_data = torch.cat((mem_data, text_data), 0)

    # Save data
    torch.save(mem_data,
               "Dataset\\PreproccessedData\\"+dataset[0]+"\\MemData\\mem.pt")


'''
Delay Pattren
'''

'''
tgt input :
    shape : [batch, batch size, length, codebook]
'''

# Prepare tgt input source
compressor = Compressor(
    max_length=melody_condition_max_length, mode="Delay")

for dataset in dir:
    audio_folder_path = "Dataset\\OriginalData\\music\\"+dataset[0]+"\\"
    audio_dirs = os.listdir(audio_folder_path)
    audio_dirs.sort()

    # For each batch
    tgt_data = None
    num = 0
    for batch in range(dataset[1]):

        # For each example
        audio_path = []
        for example in range(dataset[2]):
            audio_path.append(audio_folder_path+audio_dirs[num])
            num += 1

        # Convert to tgt input
        audio_path = [audio_path]
        audio_data = compressor.compress(audio_path)

        if tgt_data == None:
            tgt_data = audio_data

        else:
            tgt_data = torch.cat((tgt_data, audio_data), 0)

    # Save data
    torch.save(
        tgt_data,
        "Dataset\\PreproccessedData\\"+dataset[0]+"\\TgtData\\DelayPattern\\DelayTgt.pt")


'''
Parallel Pattren
'''

'''
tgt input :
    shape : [batch, batch size, length, codebook]
'''

# Prepare tgt input source
compressor = Compressor(
    max_length=melody_condition_max_length, mode="Parallel")

for dataset in dir:
    audio_folder_path = "Dataset\\OriginalData\\music\\"+dataset[0]+"\\"
    audio_dirs = os.listdir(audio_folder_path)
    audio_dirs.sort()

    # For each batch
    tgt_data = None
    num = 0
    for batch in range(dataset[1]):

        # For each example
        audio_path = []
        for example in range(dataset[2]):
            audio_path.append(audio_folder_path+audio_dirs[num])
            num += 1

        # Convert to tgt input
        audio_path = [audio_path]
        audio_data = compressor.compress(audio_path)

        if tgt_data == None:
            tgt_data = audio_data

        else:
            tgt_data = torch.cat((tgt_data, audio_data), 0)

    # Save data
    torch.save(
        tgt_data,
        "Dataset\\PreproccessedData\\"+dataset[0]+"\\TgtData\\ParallelPattern\\ParallelTgt.pt")
