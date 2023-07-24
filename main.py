import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config
from pprint import pprint
from data_process import DataProcess
from model import GPT2CausalLanguageModel

data_process = DataProcess("friends_conv")

"""
CONFIG
"""
MAX_LEN = data_process.get_max_len()
TRAIN_BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 6.25e-5
# LEARNING_RATE = 6.25e-7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_params = {
    "batch_size": TRAIN_BATCH_SIZE,
    "shuffle": True,
    "num_workers": 0
}

"""
Dataset Initialize
"""

training_set = data_process.get_dataset()
# x = len(training_set)


train_loader = DataLoader(
    training_set,
    **train_params
    )
# y = len(train_loader)

TOTAL_STEPS = len(train_loader) * EPOCHS
WARMUP_STEPS = TOTAL_STEPS // 10
"""
Model Initialize
"""
model = GPT2CausalLanguageModel()

# TODO: Linear Learning Rate Scheduler with Warmup

optimizer = torch.optim.Adam(
    model.parameters(),
    lr = LEARNING_RATE
)

def loss_fn(inputs, outputs):
    CEL = nn.CrossEntropyLoss()
    return CEL(inputs, outputs)

"""
Learning Rate Scheduler
"""
def linear_decay(step, total_step, warmup_step, lr):
    if step<warmup_step:
        return step/warmup_step * lr
    else:
        return lr * (1 - (step-warmup_step) / (total_step - warmup_step))

lambda1 = lambda step: min(1/((step+1)**0.5), (step+1) * (1/(WARMUP_STEPS**1.5)))
lambda2 = lambda step: linear_decay(step, TOTAL_STEPS, WARMUP_STEPS, LEARNING_RATE)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer= optimizer,
    lr_lambda= lambda2
)

"""
Training
"""
lrs = []
losses = []
pprint("TRAINING: ")
for epoch in range(EPOCHS):
    model.train()
    for _, data in enumerate(train_loader, 0):
        ids = data["input_ids"].to(DEVICE, dtype = torch.long)
        mask = data["mask"].to(DEVICE, dtype = torch.float)
        token_type_ids = data["token_type_ids"].to(DEVICE, dtype = torch.long)
        target = data["target_ids"].to(DEVICE, dtype = torch.long)

        output = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        output = output.permute(0,2,1)
        
        loss = loss_fn(output, target)
        if _%5000==0:
            pprint(f'Epoch: {epoch+1}, Loss:  {loss.item()}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])
        losses.append(loss.item())
        scheduler.step()
    state = {"epoch": epoch+1, "decoder": model, "optimizer": optimizer}
    torch.save(state, "checkpoints/checkpoint_" + str(epoch+1) + ".tar")
pprint("FINISHED TRAINING!")

figure , axis = plt.subplots(1,2)

axis[0].plot(range(TOTAL_STEPS),lrs, label = "Learning Rate")
axis[1].plot(range(TOTAL_STEPS),losses, label = "Losses")
# plt.legend()
plt.show()