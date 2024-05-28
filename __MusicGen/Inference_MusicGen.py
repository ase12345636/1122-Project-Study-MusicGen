import os
import torch

from pathlib import Path

from MusicGenModel.MusicGen.MusicGen import MusicGen
from MusicGenModel.Optimizer.Performance_Metrics import Performance_Metrics
from MusicGenModel.Compressor.Compressor import Compressor
from Config.Config import mode_table
from Config.Config import device, delay_pattern_ntoken, d_model, nheads, nlayer, d_hid, dropout, PATH, validation_max_num_batch_size


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
model = MusicGen(
    tgt_ntoken=delay_pattern_ntoken,
    d_model=d_model,
    nhead=nheads,
    nlayer=nlayer,
    d_hid=d_hid,
    dropout=dropout,
    melody_condition_max_length=melody_condition_max_length).to(device)
# model.load_state_dict(torch.load(Path(PATH+mode+"Model_Last.h5")))
model.eval()

# Load performance metrics
FAD = Performance_Metrics()

# Load compressor
compressor = Compressor(mode=mode)

# Load validation path
mem_validation_folder_path = Path(
    "Dataset/PreproccessedData/validation/MemData")
if os.path.isdir(mem_validation_folder_path/".ipynb_checkpoints"):
    (mem_validation_folder_path/".ipynb_checkpoints").rmdir()
mem_validation_dirs = os.listdir(mem_validation_folder_path)
mem_validation_dirs.sort()

# Start inference
print("Inference...")

with torch.no_grad():

    # For each data file part
    for file_part in range(len(mem_validation_dirs)):

        mem_validation_data = torch.load(mem_validation_folder_path/mem_validation_dirs[file_part],
                                         map_location=device, weights_only=True)

        # For each batch
        for example in range(len(mem_validation_data[0])):

            # Inference
            inferenced_mem_validation_data = torch.reshape(
                mem_validation_data[0, example, :], (1, text_condition_max_length, -1))

            # Generate new audio token
            output = model.generation(
                inferenced_mem_validation_data, mode=mode)

            # Decompress new audio token to new audio
            compressor.decompress(
                output, Path("Dataset/InferenceData/validation/"+mode+"Pattern/"+mode+"Output_"+str(file_part*validation_max_num_batch_size+example)+".wav"))

    # Delete unnecessary folder
    if os.path.isdir(Path("Dataset/OriginalData/music/validation/.ipynb_checkpoints")):
        (Path("Dataset/OriginalData/music/validation/.ipynb_checkpoints")).rmdir()

    if os.path.isdir(Path("Dataset/InferenceData/validation/"+mode+"Pattern/.ipynb_checkpoints")):
        (Path("Dataset/InferenceData/validation/" +
         mode+"Pattern/.ipynb_checkpoints")).rmdir()

    # Compute validation fad
    fad_score = FAD.FAD(Path("Dataset/OriginalData/music/validation"),
                        Path("Dataset/InferenceData/validation/"+mode+"Pattern"))

    print(f"Validetion FAD: {fad_score}")
