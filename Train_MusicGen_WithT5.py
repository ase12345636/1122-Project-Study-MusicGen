import os
import random
import matplotlib.pyplot as plt
import numpy as np

import torch
import copy

from pathlib import Path

from torch.cuda.amp import autocast as autocast

from torchsummary import summary
from MusicGenModel.MusicGen.MusicGen_WithT5 import MusicGen_WithT5
from MusicGenModel.Optimizer.Model_Ema import Model_Ema
from MusicGenModel.Optimizer.Loss_Function import Loss_Function
from MusicGenModel.Optimizer.Optimizer import Optimizer
from MusicGenModel.Optimizer.Performance_Metrics import Performance_Metrics
from MusicGenModel.Compressor.Compressor import Compressor
from Config.Config_800M  import mode_table
from Config.Config_800M  import device, d_model, nheads, nlayer, d_hid, dropout, num_epoch, batch_multiplier, lr, betas, eps, PATH, loss_function_max_length


'''
Change mode to delay pattern mode or parallel pattern mode :

    0 -> delay pattern mode
    1 -> parallel pattern mode
'''
mode_list = mode_table[0]

# Load mode, max length
mode = mode_list[0]
text_condition_max_length = mode_list[1]
melody_condition_max_length = mode_list[2]
ntoken = mode_list[3]

# Load model
model = MusicGen_WithT5(
    tgt_ntoken=ntoken,
    d_model=d_model,
    nhead=nheads,
    nlayer=nlayer,
    d_hid=d_hid,
    dropout=dropout,
    melody_condition_max_length=melody_condition_max_length).to(device)
# model.load_state_dict(torch.load(Path(PATH+mode+"Model_Best.h5")))
# summary(model)

# Load ema model
model_ema = Model_Ema(model)

# Load optimizer
optimizer, sechdualer = Optimizer(model, lr, betas, eps)

# Load performance metrics
FAD = Performance_Metrics()

# Load compressor
compressor = Compressor(mode=mode)

# Load training path
mem_train_folder_path = Path(
    "Dataset/PreproccessedData/train/MemData/TokenIDs")
if os.path.isdir(mem_train_folder_path/".ipynb_checkpoints"):
    (mem_train_folder_path/".ipynb_checkpoints").rmdir()
mem_train_dirs = os.listdir(mem_train_folder_path)
mem_train_dirs.sort()

tgt_train_folder_path = Path(
    "Dataset/PreproccessedData/train/TgtData/"+mode+"Pattern")
if os.path.isdir(tgt_train_folder_path / ".ipynb_checkpoints"):
    (tgt_train_folder_path / ".ipynb_checkpoints").rmdir()
tgt_train_dirs = os.listdir(tgt_train_folder_path)
tgt_train_dirs.sort()

# Load validation path
mem_validation_folder_path = Path(
    "Dataset/PreproccessedData/validation/MemData/TokenIDs")
if os.path.isdir(mem_validation_folder_path/".ipynb_checkpoints"):
    (mem_validation_folder_path/".ipynb_checkpoints").rmdir()
mem_validation_dirs = os.listdir(mem_validation_folder_path)
mem_validation_dirs.sort()

tgt_validation_folder_path = Path(
    "Dataset/PreproccessedData/validation/TgtData/"+mode+"Pattern")
if os.path.isdir(tgt_validation_folder_path / ".ipynb_checkpoints"):
    (tgt_validation_folder_path / ".ipynb_checkpoints").rmdir()
tgt_validation_dirs = os.listdir(tgt_validation_folder_path)
tgt_validation_dirs.sort()

# Records
iteration = 0
count = batch_multiplier
last_loss = float("inf")
min_loss = float("inf")
total_loss = 0.0
iteration_history = []
loss_history = []
lr_history = []

# Start training
model.train()

# For each epoch
print("Training...")

for epoch in range(num_epoch):

    # For each data file part
    optimizer.zero_grad()

    # For training data
    shuffle_data = [i for i in range(len(mem_train_dirs))]
    random.shuffle(shuffle_data)

    for file_part in range(len(shuffle_data)):

        selected_data = shuffle_data[file_part]
        mem_train_data = torch.load(
            mem_train_folder_path/mem_train_dirs[selected_data], map_location=device)

        tgt_train_data = torch.load(
            tgt_train_folder_path/tgt_train_dirs[selected_data], map_location=device)

        # For each batch
        for batch in range(len(tgt_train_data)):

            with autocast():

                # Load training data and ground truth
                mem = mem_train_data[batch, :14, :]
                tgt = tgt_train_data[batch, :14, :, :-1]
                tgt_gt = tgt_train_data[batch, :14, :, 1:]

                # Get output
                prediction = model(mem, tgt)

                # Compute loss with 4 codebooks
                loss = Loss_Function(prediction=prediction,
                                     tgt_gt=tgt_gt,
                                     max_length=loss_function_max_length,
                                     mode=mode)

            # Gradient accumulation
            loss /= float(batch_multiplier)
            loss.backward()

            # Update recodes
            count -= 1
            total_loss += loss.item()

            # Update parameter
            if count == 0 or file_part+1 == len(mem_train_dirs):

                # Update model's parameter
                optimizer.step()
                optimizer.zero_grad()

                # Update model_ema's parameter
                model_ema.update(model)

                # Print loss for each iteration
                iteration += 1
                iteration_history.append(iteration)
                loss_history.append(total_loss)
                lr_history.append(sechdualer.get_lr())
                print(f"Iteration: {iteration}, Loss: {total_loss}")
                print(sechdualer.get_lr())

                # Update recodes
                count = batch_multiplier
                last_loss = total_loss
                min_loss = min(min_loss, last_loss)
                total_loss = 0.0

                print("Save...")

                # Save best model
                if min_loss == last_loss or iteration == 1:
                    torch.save(model.state_dict(), Path(
                        PATH+mode+"Model_WithT5_Best.h5"))
                    torch.save(model_ema.model.state_dict(), Path(
                        PATH+mode+"Model_WithT5_Ema_Best.h5"))

    # # For validation data
    # shuffle_data = [i for i in range(len(mem_validation_dirs))]
    # random.shuffle(shuffle_data)

    # for file_part in range(len(shuffle_data)):

    #     selected_data = shuffle_data[file_part]

    #     mem_validation_data = torch.load(
    #         mem_validation_folder_path/mem_validation_dirs[selected_data], map_location=device)

    #     tgt_validation_data = torch.load(
    #         tgt_validation_folder_path/tgt_validation_dirs[selected_data], map_location=device)

    #     # For each batch
    #     for batch in range(len(tgt_validation_data)):

    #         with autocast():

    #             # Load validationing data and ground truth
    #             mem = mem_validation_data[batch, :14, :]
    #             tgt = tgt_validation_data[batch, :14, :, :-1]
    #             tgt_gt = tgt_validation_data[batch, :14, :, 1:]

    #             # Get output
    #             prediction = model(mem, tgt)

    #             # Compute loss with 4 codebooks
    #             loss = Loss_Function(prediction=prediction,
    #                                  tgt_gt=tgt_gt,
    #                                  max_length=loss_function_max_length,
    #                                  mode=mode)

    #         # Gradient accumulation
    #         loss /= float(batch_multiplier)
    #         loss.backward()

    #         # Update recodes
    #         count -= 1
    #         total_loss += loss.item()

    #         # Update parameter
    #         if count == 0 or file_part+1 == len(mem_validation_dirs):

    #             # Update model's parameter
    #             optimizer.step()
    #             optimizer.zero_grad()

    #             # Update model_ema's parameter
    #             model_ema.update(model)

    #             # Print loss for each iteration
    #             iteration += 1
    #             iteration_history.append(iteration)
    #             loss_history.append(total_loss)
    #             lr_history.append(sechdualer.get_lr())
    #             print(f"Iteration: {iteration}, Loss: {total_loss}")
    #             print(sechdualer.get_lr())

    #             # Update recodes
    #             count = batch_multiplier
    #             last_loss = total_loss
    #             min_loss = min(min_loss, last_loss)
    #             total_loss = 0.0

    #             print("Save...")

    #             # Save best model
    #             if min_loss == last_loss or iteration == 1:
    #                 torch.save(model.state_dict(), Path(
    #                     PATH+mode+"Model_WithT5_Best.h5"))
    #                 torch.save(model_ema.model.state_dict(), Path(
    #                     PATH+mode+"Model_WithT5_Ema_Best.h5"))

    # Save last model
    torch.save(model.state_dict(), Path(
        PATH+mode+"Model_WithT5_Last.h5"))
    torch.save(model_ema.model.state_dict(), Path(
        PATH+mode+"Model_WithT5_Ema_Last.h5"))

    sechdualer.step()

print(f"Finish training\nMin_loss = {min_loss}\nLast_loss = {last_loss}")

plt.plot(np.array(iteration_history),
         np.array(loss_history), label='training loss')
plt.savefig(Path("loss_history.png"))
plt.show()

plt.plot(np.array(iteration_history),
         np.array(lr_history), label='training loss')
plt.savefig(Path("lr_history.png"))
plt.show()
