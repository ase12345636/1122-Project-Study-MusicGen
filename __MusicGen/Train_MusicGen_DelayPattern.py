import torch

from torch.cuda.amp import autocast as autocast

from torchsummary import summary
from MusicGenModel.MusicGen.MusicGen import MusicGen
from MusicGenModel.Optimizer.Loss_Function import Loss_Function
from MusicGenModel.Optimizer.Optimizer import Optimizer
from Config.Config import device, delay_pattern_ntoken, d_model, nheads, nlayer, d_hid, dropout, lr, betas, eps, PATH, melody_condition_max_length


mode = "Delay"

# Load model
transformer = MusicGen(
    tgt_ntoken=delay_pattern_ntoken,
    d_model=d_model,
    nhead=nheads,
    nlayer=nlayer,
    d_hid=d_hid,
    dropout=dropout,
    melody_condition_max_length=melody_condition_max_length+3).to(device)
# transformer.load_state_dict(torch.load(PATH))
summary(transformer)

# Load optimizer
optimizer = Optimizer(transformer, lr, betas, eps)

# Load data
mem_data = torch.load('Dataset\\PreproccessedData\\train\\MemData\\mem.pt',
                      map_location=device, weights_only=True)
tgt_data = torch.load('Dataset\\PreproccessedData\\train\\TgtData\\'+mode+'Pattern\\'+mode+'Tgt.pt',
                      map_location=device, weights_only=True)

# Start training
transformer.train()

# For each eopch
iteration = 0
for epoch in range(10):

    # For each batch
    for batch in range(len(tgt_data)):
        optimizer.zero_grad()

        with autocast():
            # Load training data and ground truth
            mem = mem_data[batch, :, :]
            tgt = tgt_data[batch, :, :, :-1]
            tgt_gt = tgt_data[batch, :, :, 1:]

            prediction = transformer(
                mem, tgt)

            # Compute loss with 4 codebooks
            loss = Loss_Function(prediction=prediction,
                                 tgt_gt=tgt_gt,
                                 max_length=melody_condition_max_length,
                                 mode=mode)

        # Update parameters
        loss.backward()

        optimizer.step()

        iteration += 1
        print(f"Iteration: {iteration}, Loss: {loss.item()}")

# Save model
torch.save(transformer.state_dict(), PATH+"DelayModel.h5")
