import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config
from pprint import pprint
from data_process import DataProcess
from model import GPT2CausalLanguageModel

data_process = DataProcess("DM_Kaiyzxn")
max_len = data_process.get_max_len()
tokenizer = data_process.tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load("checkpoints/m2_checkpoint_5.tar")
model = checkpoint["decoder"]
# pprint(model)

sample_test = "<|endoftext|> How are you doing?"
sample_test = tokenizer.encode_plus(
    sample_test,
    None,
    return_token_type_ids = True
    )

ids = torch.tensor(sample_test["input_ids"], dtype= torch.long).to(device)
mask = torch.tensor(sample_test["attention_mask"], dtype= torch.float).to(device)
token_type = torch.tensor(sample_test["token_type_ids"], dtype= torch.long).to(device)


# pprint(ids)
# pprint(ids.shape)

# pprint(mask)
# pprint(mask.shape)

# pprint(token_type)
# pprint(token_type.shape)
# pprint(tokenizer.decode(ids))


with torch.no_grad():
    for i in range(0,15):
        output = model(ids, mask, token_type)
        output = output[-1,:]
        output = torch.softmax(output, dim = -1)
        # pprint(output)
        # pprint(output.shape)
        # max_prob = torch.argmax(output)
        # pprint(max_prob)
        sorted_ids = torch.argsort(output, dim=-1, descending=True)
        # pprint(sorted_ids[1])
        pprint(sorted_ids)
        ids = torch.concat([ids, sorted_ids[0].unsqueeze(0)], dim= -1)
        # pprint(ids)
        result = tokenizer.decode(ids)
        print(result)
        encoded = tokenizer.encode_plus(
            result,
            None,
            return_token_type_ids = True
            )
        ids = torch.tensor(encoded["input_ids"], dtype= torch.long).to(device)
        mask = torch.tensor(encoded["attention_mask"], dtype= torch.float).to(device)
        token_type = torch.tensor(encoded["token_type_ids"], dtype= torch.long).to(device)
