import torch

from torch.cuda.amp import autocast as autocast

from torchsummary import summary
from MusicGenModel.MusicGen.MusicGen_ParallelPattern import MusicGen_ParallelPattern
from MusicGenModel.Optimizer.Loss_Function import Loss_Function
from MusicGenModel.Optimizer.Optimizer import Optimizer
from Config.Config import device, parallel_pattern_ntoken, d_model, nheads, nlayer, d_hid, dropout, ignore_index, lr, betas, eps, PATH, melody_condition_max_length


# Load model
transformer = MusicGen_ParallelPattern(
    tgt_ntoken=parallel_pattern_ntoken,
    d_model=d_model,
    nhead=nheads,
    nlayer=nlayer,
    d_hid=d_hid,
    dropout=dropout,
    melody_condition_max_length=melody_condition_max_length).to(device)
# transformer.load_state_dict(torch.load(PATH))
summary(transformer)

# Load loss function and optimizer
criterion = Loss_Function(ignore_index)
optimizer = Optimizer(transformer, lr, betas, eps)

# Load data
mem_data = torch.load('Dataset\\PreproccessedData\\train\\mem.pt',
                      map_location=device, weights_only=True)
tgt_data = torch.load('Dataset\\PreproccessedData\\train\\tgt.pt',
                      map_location=device, weights_only=True)

# Start training
transformer.train()

# For each eopch
iteration = 0
for epoch in range(1000):

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
            loss = 0.0
            for example in range(len(prediction)):
                for codebook in range(4):
                    loss += criterion(prediction[example, codebook, :, :].contiguous().view(-1, parallel_pattern_ntoken-1),
                                      tgt_gt[example, codebook, :].contiguous().view(-1))

        # Update parameters
        loss /= float(len(prediction))
        loss.backward()

        optimizer.step()

        iteration += 1
        print(f"Iteration: {iteration}, Loss: {loss.item()}")

# Save model
torch.save(transformer.state_dict(), PATH)
