import os
import torch

from pathlib import Path

from MusicGenModel.MusicGen.MusicGen import MusicGen
from MusicGenModel.Optimizer.Performance_Metrics import Performance_Metrics
from MusicGenModel.Compressor.Compressor import Compressor
from Config.Config_100M import mode_table
from Config.Config_100M import device, delay_pattern_ntoken, d_model, nheads, nlayer, d_hid, dropout, PATH, test_max_num_batch_size


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
model.load_state_dict(torch.load(Path(PATH+mode+"Model_Ema_Best_First300CleanData_100M.h5")))
model.eval()

# Load performance metrics
FAD = Performance_Metrics()

# Load compressor
compressor = Compressor(mode=mode)

# Load test path
mem_test_folder_path = Path(
    "Dataset/PreproccessedData/Demo/MemData/Mem")
if os.path.isdir(mem_test_folder_path/".ipynb_checkpoints"):
    (mem_test_folder_path/".ipynb_checkpoints").rmdir()
mem_test_dirs = os.listdir(mem_test_folder_path)
mem_test_dirs.sort()

with torch.no_grad():

    # Start inference
    print("Inference...")

    # For each data file part
    for file_part in range(len(mem_test_dirs)):

        mem_test_data = torch.load(mem_test_folder_path/mem_test_dirs[file_part],
                                   map_location=device, weights_only=True)

        # For each batch
        for example in range(len(mem_test_data[0])):

            # Inference
            inferenced_mem_test_data = torch.reshape(
                mem_test_data[0, example, :], (1, text_condition_max_length, -1))

            # Generate new audio token
            output = model.generation(
                inferenced_mem_test_data, mode=mode)

            # Decompress new audio token to new audio
            compressor.decompress(
                output, Path("Dataset/InferenceData/Demo/"+mode+"Pattern/"+mode+"Output_"+str(file_part*test_max_num_batch_size+example)+".wav"))

    # # Start inference
    # print("Compute FAD Score...")

    # # Delete unnecessary folder
    # if os.path.isdir(Path("Dataset/OriginalData/music/test/TgtAudioData/.ipynb_checkpoints")):
    #     (Path("Dataset/OriginalData/music/test/.ipynb_checkpoints")).rmdir()

    # if os.path.isdir(Path("Dataset/InferenceData/test/"+mode+"Pattern/.ipynb_checkpoints")):
    #     (Path("Dataset/InferenceData/test/" +
    #      mode+"Pattern/.ipynb_checkpoints")).rmdir()

    # # Compute test fad
    # fad_score = FAD.FAD(Path("Dataset/PreproccessedData/test/TgtAudioData"),
    #                     Path("Dataset/InferenceData/test/"+mode+"Pattern"))

    # print(f"Test FAD: {fad_score}")
