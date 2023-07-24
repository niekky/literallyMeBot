import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model
from pprint import pprint
from data_process import DataProcess
import matplotlib.pyplot as plt

class GPT2CausalLanguageModel(nn.Module):
    def __init__(self):
        super(GPT2CausalLanguageModel, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # device = "cpu"
        self.config = GPT2Config()
        self.gpt2 = GPT2Model.from_pretrained("gpt2", return_dict = False).to(device)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False).to(device)
    
    def forward(self, input_ids, mask, token_type_ids):
        out = self.gpt2(
            input_ids,
            None,
            mask,
            token_type_ids
        )
        out = out[0]
        out = self.dropout(out)
        out = self.linear(out)
        return out

def loss_fn(inputs, outputs):
    CEL = nn.CrossEntropyLoss()
    return CEL(inputs, outputs)


"""
Testing
"""
# model = GPT2CausalLanguageModel()
# pprint(GPT2Config())
# # # pprint(model)
# data_process = DataProcess("friends_conv")
# sample = data_process.get_dataset()
# sample = sample.__getitem__(1)
# pprint(sample)
# ids = sample["input_ids"].unsqueeze(0)
# mask = sample["mask"].unsqueeze(0)
# token_type = sample["token_type_ids"].unsqueeze(0)
# target = sample["target_ids"].unsqueeze(0)
# out = model(ids, mask, token_type)

# # out = out.view(-1, out.shape[-1])
# # target = target.view(-1)
# out = out.permute(0,2,1)
# pprint(out.shape)
# pprint(target)
# pprint(target.shape)
# pprint(loss_fn(out, target))

## Test LearningRate Scheduler
# lr = 6.25e-5
# sample_epoch = 5
# total_steps = len(sample) * sample_epoch
# warmup_steps = total_steps // 10

# linear = torch.nn.Linear(2, 1)
# optimizer = torch.optim.Adam(linear.parameters(), lr)
# lambda1 = lambda step: min(1.0, step/warmup_steps)
# lambda2 = lambda step: 0.65 ** step
# lambda3 = lambda step: (1 - step/total_steps) ** 0.5 * (step < warmup_steps) + (step >= warmup_steps) * 0.5
# lambda4 = lambda step: min(1/((step+1)**0.5), (step+1) * (1/(warmup_steps**1.5)))

# def linear_decay(step, total_step, warmup_step, lr):
#     if step<warmup_step:
#         return step/warmup_step * lr
#     else:
#         return lr * (1 - (step-warmup_step) / (total_step - warmup_step))
    
# lambda5 = lambda step: linear_decay(step, total_steps, warmup_steps, lr)

# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer= optimizer, lr_lambda = lambda4)

# lrs = []

# for i in range(total_steps):
#     optimizer.step()
#     lrs.append(optimizer.param_groups[0]["lr"])
#     scheduler.step()

# plt.plot(range(total_steps),lrs)
# plt.show()
# print("PLOTTED")