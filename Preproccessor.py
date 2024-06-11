import os
import math

import json
import torch

from pathlib import Path

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
                word_dropout=0.0,
                max_length=text_condition_max_length)

for dataset in dir:

    type_dataset = dataset[0]
    max_batch = dataset[1]
    max_batch_size = dataset[2]

    print("Proccess mem data : "+type_dataset+" dataset")

    # Load data file
    text_folder_path = Path("Dataset/OriginalData/json/"+type_dataset)
    text_dirs = os.listdir(text_folder_path)
    text_dirs.sort()

    # For each dataset part
    num = 0
    for part in range(math.ceil(len(text_dirs)/max_batch/max_batch_size)):

        # For each batch
        mem_data = None
        for batch in range(max_batch):

            if num >= len(text_dirs):
                break

            # For each example
            text_path = []
            for example in range(max_batch_size):

                if num >= len(text_dirs):
                    break

                json_file_path = text_folder_path/text_dirs[num]
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
        print(mem_data.shape)
        torch.save(mem_data,
                   Path("Dataset/PreproccessedData/"+type_dataset+"/MemData/Mem/mem_part"+str(part)+".pt"))

# '''
# mem ids input :
#     shape : [batch, batch size, length]
# '''

# # Prepare mem input source
# t5 = Encoder_T5(finetune=False,
#                 word_dropout=word_dropout,
#                 max_length=text_condition_max_length)

# for dataset in dir:

#     type_dataset = dataset[0]
#     max_batch = dataset[1]
#     max_batch_size = dataset[2]

#     print("Proccess mem data : "+type_dataset+" dataset")

#     # Load data file
#     text_folder_path = Path("Dataset/OriginalData/json/"+type_dataset)
#     text_dirs = os.listdir(text_folder_path)
#     text_dirs.sort()

#     # For each dataset part
#     num = 0
#     for part in range(math.ceil(len(text_dirs)/max_batch/max_batch_size)):

#         # For each batch
#         mem_data = None
#         for batch in range(max_batch):

#             if num >= len(text_dirs):
#                 break

#             # For each example
#             text_path = []
#             for example in range(max_batch_size):

#                 if num >= len(text_dirs):
#                     break

#                 json_file_path = text_folder_path/text_dirs[num]
#                 json_data = json.load(open(json_file_path))
#                 text_path.append(json_data["caption"])
#                 num += 1

#             # Convert to mem input
#             text_path = [[text_path]]
#             text_data = t5(text_path[0], True)

#             if mem_data == None:
#                 mem_data = text_data

#             else:
#                 mem_data = torch.cat((mem_data, text_data), 0)

#         # Save data
#         print(mem_data.shape)
#         torch.save(mem_data,
#                    Path("Dataset/PreproccessedData/"+type_dataset+"/MemData/TokenIDs/mem_part"+str(part)+".pt"))

# '''
# Delay Pattren

# tgt input :
#     shape : [batch, batch size, length, codebook]
# '''

# # Prepare tgt input source
# compressor = Compressor(
#     max_length=melody_condition_max_length, mode="Delay")

# for dataset in dir:

#     type_dataset = dataset[0]
#     max_batch = dataset[1]
#     max_batch_size = dataset[2]

#     print("Proccess tgt data for delay pattern : "+type_dataset+" dataset")

#     # Load data file
#     audio_folder_path = Path("Dataset/OriginalData/music/"+type_dataset)
#     audio_dirs = os.listdir(audio_folder_path)
#     audio_dirs.sort()

#     # For each dataset part
#     num = 0
#     for part in range(math.ceil(len(audio_dirs)/max_batch/max_batch_size)):

#         # For each batch
#         tgt_data = None
#         for batch in range(max_batch):

#             if num >= len(audio_dirs):
#                 break

#             # For each example
#             audio_path = []
#             for example in range(max_batch_size):

#                 if num >= len(audio_dirs):
#                     break
#                 audio_path.append((audio_folder_path/audio_dirs[num]))
#                 num += 1

#             # Convert to tgt input
#             audio_path = [audio_path]
#             audio_data = compressor.compress(audio_path)

#             if tgt_data == None:
#                 tgt_data = audio_data

#             else:
#                 tgt_data = torch.cat((tgt_data, audio_data), 0)

#         # Save data
#         print(tgt_data.shape)
#         torch.save(
#             tgt_data,
#             Path("Dataset/PreproccessedData/"+type_dataset+"/TgtData/DelayPattern/DelayTgt_part"+str(part)+".pt"))


# '''
# Parallel Pattren

# tgt input :
#     shape : [batch, batch size, length, codebook]
# '''

# # Prepare tgt input source
# compressor = Compressor(
#     max_length=melody_condition_max_length, mode="Parallel")

# for dataset in dir:

#     type_dataset = dataset[0]
#     max_batch = dataset[1]
#     max_batch_size = dataset[2]

#     print("Proccess mem data for parallel : "+type_dataset+" dataset")

#     # Load data file
#     audio_folder_path = Path("Dataset/OriginalData/music/"+type_dataset)
#     audio_dirs = os.listdir(audio_folder_path)
#     audio_dirs.sort()

#     # For each dataset part
#     num = 0
#     for part in range(math.ceil(len(audio_dirs)/max_batch/max_batch_size)):

#         # For each batch
#         tgt_data = None
#         for batch in range(max_batch):

#             if num >= len(audio_dirs):
#                 break

#             # For each example
#             audio_path = []
#             for example in range(max_batch_size):

#                 if num >= len(audio_dirs):
#                     break

#                 audio_path.append(audio_folder_path/audio_dirs[num])
#                 num += 1

#             # Convert to tgt input
#             audio_path = [audio_path]
#             audio_data = compressor.compress(audio_path)

#             if tgt_data == None:
#                 tgt_data = audio_data

#             else:
#                 tgt_data = torch.cat((tgt_data, audio_data), 0)

#         # Save data
#         print(tgt_data.shape)
#         torch.save(
#             tgt_data,
#             Path("Dataset/PreproccessedData/"+type_dataset+"/TgtData/ParallelPattern/ParallelTgt_part"+str(part)+".pt"))


# '''
# tgt audio : 5-sec audio with 32k-sample_rate
# '''

# # Prepare tgt audio
# compressor = Compressor(
#     max_length=melody_condition_max_length)

# type_dataset = dir[1][0]
# max_batch = dir[1][1]
# max_batch_size = dir[1][2]

# print("Proccess tgt audio : "+type_dataset+" dataset")

# # Load data file
# audio_folder_path = Path("Dataset/OriginalData/music/"+type_dataset)
# audio_dirs = os.listdir(audio_folder_path)
# audio_dirs.sort()

# # For each dataset part
# num = 0
# for audio_file in audio_dirs:

#     compressor.resample_audio(audio_folder_path/audio_file, Path(
#         "Dataset/PreproccessedData/"+type_dataset+"/TgtAudioData/"+audio_file))
