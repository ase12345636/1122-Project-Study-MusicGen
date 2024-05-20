import torch

from torch.cuda.amp import autocast as autocast

from torchsummary import summary
from MusicGenModel.MusicGen.MusicGen import MusicGen
from MusicGenModel.Optimizer.Loss_Function import Loss_Function
from MusicGenModel.Optimizer.Optimizer import Optimizer
from MusicGenModel.Optimizer.Performance_Metrics import Performance_Metrics
from MusicGenModel.Compressor.Compressor import Compressor
from Config.Config import mode_table
from Config.Config import device, delay_pattern_ntoken, d_model, nheads, nlayer, d_hid, dropout, epoch_num, lr, betas, eps, PATH, loss_function_max_length
from Config.Config import validation_dataset_batch_size


'''
Change mode to delay pattern mode or parallel pattern mode :

    0 -> delay pattern mode
    1 ->parallel pattern mode
'''
mode_list = mode_table[1]

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
# transformer.load_state_dict(torch.load(PATH))
summary(transformer)

# Load optimizer
optimizer = Optimizer(transformer, lr, betas, eps)

FAD = Performance_Metrics()

# Load compressor
compressor = Compressor(mode=mode)

# Load data
mem_train_data = torch.load('Dataset\\PreproccessedData\\train\\MemData\\mem.pt',
                            map_location=device, weights_only=True)
tgt_data = torch.load('Dataset\\PreproccessedData\\train\\TgtData\\'+mode+'Pattern\\'+mode+'Tgt.pt',
                      map_location=device, weights_only=True)
mem_validation_data = torch.load('Dataset\\PreproccessedData\\validation\\MemData\\mem.pt',
                                 map_location=device, weights_only=True)

# Start training
transformer.train()

# For each eopch
iteration = 0
for epoch in range(epoch_num):

    # For each batch
    for batch in range(len(tgt_data)):
        optimizer.zero_grad()

        with autocast():
            # Load training data and ground truth
            mem = mem_train_data[batch, :, :]
            tgt = tgt_data[batch, :, :, :-1]
            tgt_gt = tgt_data[batch, :, :, 1:]

            prediction = transformer(
                mem, tgt)

            # Compute loss with 4 codebooks
            loss = Loss_Function(prediction=prediction,
                                 tgt_gt=tgt_gt,
                                 max_length=loss_function_max_length,
                                 mode=mode)

        # Update parameters
        loss.backward()
        optimizer.step()

        # Print loss for each iteration
        iteration += 1
        print(f"Iteration: {iteration}, Loss: {loss.item()}")

        # # Compute validation fad
        # # Start inference
        # with torch.no_grad():
        #     for example in range(validation_dataset_batch_size):
        #         inferenced_mem_validation_data = torch.reshape(
        #             mem_validation_data[0, example, :], (1, text_condition_max_length, -1))

        #         output = transformer.generation(inferenced_mem_validation_data)

        #         compressor.decompress(
        #             output, "Dataset\\InferenceData\\validation\\"+mode+"Pattern\\Test"+mode+"Output_"+str(example)+".wav")

        #         fad_score = FAD.FAD("Dataset\\OriginalData\\music\\validation",
        #                             "Dataset\\InferenceData\\validation\\"+mode+"Pattern")

        #         print(f"Validetion FAD: {fad_score}")


# Save model
torch.save(transformer.state_dict(), PATH+mode+"Model.h5")
