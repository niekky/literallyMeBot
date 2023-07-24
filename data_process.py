import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
from pprint import pprint
import json

class DataProcess():
    def __init__(self, name):
        self.data_dict = self.load_json(name)
        # self.max_len = self.find_max_len(self.data_dict)
        self.max_len = 40
        # self.pairs = self.convert_to_ques_ans(self.data_dict)
        # self.pairs = self.convert_to_one_sentence_conv(self.data_dict)
        self.pairs = self.load_json("aug_pairs2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.add_special_tokens(
            {
                'pad_token': '<|endoftext|>',
                "eos_token": "<|endoftext|>",
                "bos_token": "<|endoftext|>"
            }
        )

    def load_json(self, name):
        with open(f"data\{name}.json") as f:
            data_dict = json.load(f)
        return data_dict

    def convert_to_ques_ans(self, dataset): #Only works for encoder2decoder models
        q_a = []
        target_bot = ".nick#2667"
        i = 0
        for i in range(0, len(dataset), 2):
            if list(dataset[i].keys())[0] != target_bot:
                pair = []
                pair.append(list(dataset[i].values())[0])   #other user
                pair.append(list(dataset[i+1].values())[0]) #target   
            q_a.append(pair)
        return q_a

    def quote(self, sentence):
        return "\"" + sentence + "\""

    def convert_to_one_sentence_conv(self, dataset): #Only works for decoder only models
        target_bot = ".nick#2667"
        i = 0
        result = []
        for i in range(0, len(dataset), 2):
            if list(dataset[i].keys())[0] != target_bot and (len(dataset) - i) >= 2 :
                other_user = list(dataset[i].values())[0]
                target_user = list(dataset[i+1].values())[0]
                # sentence = self.quote(other_user) + ". i replied: " + self.quote(target_user)
                sentence = other_user + " i replied: " + target_user
                result.append([sentence, target_user])

        return result

    def find_max_len(self, dataset):
        first_sentence = list(dataset[0].values())[0]
        max_len = len(first_sentence)

        for index, i in enumerate(dataset):
            curr_len = len(list(i.values())[0])
            if curr_len > max_len:
                max_len_ids = index
                max_len = curr_len
        return max_len_ids, max_len

    def get_dataset(self):
        return CustomDataset(
            pairs= self.pairs,
            tokenizer= self.tokenizer,
            max_len= self.max_len
        )

    def get_max_len(self):
        return self.max_len

    def get_pairs(self):
        return self.pairs

    def get_data_dict(self):
        return self.data_dict


class CustomDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        ques = self.pairs[index][0]
        ans = self.pairs[index][1]

        ques_encoded = self.encode_input(
            sentence= ques,
            tokenizer= self.tokenizer,
            max_len= self.max_len
        )
        ids = ques_encoded["input_ids"]
        mask = ques_encoded["attention_mask"]
        token_type_ids = ques_encoded["token_type_ids"]

        ans_encoded = self.encode_target(
            sentence= ans,
            tokenizer= self.tokenizer,
            max_len= self.max_len
        )
        target_ids = ans_encoded["input_ids"]
        target_mask = ans_encoded["attention_mask"]
        target_token_type_ids = ans_encoded["token_type_ids"]

        return {
            "input_ids": torch.tensor(ids, dtype= torch.long),
            "mask": torch.tensor(mask, dtype= torch.float),
            "token_type_ids": torch.tensor(token_type_ids, dtype= torch.long),
            "target_ids": torch.tensor(target_ids, dtype= torch.long),
            "target_mask": torch.tensor(target_mask, dtype= torch.float),
            "target_token_type_ids": torch.tensor(target_token_type_ids, dtype= torch.long)
        }

    def encode_input(self, sentence, tokenizer, max_len):
        sentence = "<|endoftext|> " + sentence
        encoded = tokenizer.encode_plus(
            sentence,
            None,
            max_length = max_len,
            return_token_type_ids = True)
        encoded["input_ids"] = encoded["input_ids"] + [50256] * (max_len - len(encoded["input_ids"]))
        encoded["attention_mask"] = encoded["attention_mask"] + [0] * (max_len - len(encoded["attention_mask"]))
        encoded["token_type_ids"] = encoded["token_type_ids"] + [0] * (max_len - len(encoded["token_type_ids"]))
        return encoded

    def encode_target(self, sentence, tokenizer, max_len):
        sentence = sentence + " <|endoftext|>"
        encoded = tokenizer.encode_plus(
            sentence,
            None,
            max_length = max_len,
            return_token_type_ids = True)
        encoded["input_ids"] = encoded["input_ids"] + [50256] * (max_len - len(encoded["input_ids"]))
        encoded["attention_mask"] = encoded["attention_mask"] + [0] * (max_len - len(encoded["attention_mask"]))
        encoded["token_type_ids"] = encoded["token_type_ids"] + [0] * (max_len - len(encoded["token_type_ids"]))
        return encoded

"""
Process data with GPT2
"""
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# sample = "<|endoftext|> lmao ay you lose lol <|endoftext|>"
# encoded = tokenizer.encode_plus(
#     sample,
#     add_special_tokens = True,
#     return_token_type_ids = True    
#     )
# pprint(encoded)
# decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens = False)
# pprint(decoded)

"""
Write sentence group
"""
# data_test = sentence_group(raw, data_length)

# with open("data\group_CKH.json", "w") as f:
#     json.dump(data_test, f)

# with open("data\group_CKH.json", "r") as f:
#     sample = json.load(f)

# # pprint(sample[10:20])
# pprint(list(sample[0].values()))
"""
Input: everyone else message
Target: .nick message
"""


# data_process = DataProcess("DM_Kaiyzxn")
# pprint(data_process.get_max_len())
# sample = data_process.get_data_dict()
# pprint(sample[:5])

# sample = data_process.get_pairs()
# pprint(sample[:5])


# sample_dataset = data_process.get_dataset()
# sample_dataset = sample_dataset.__getitem__(0)
# pprint(sample_dataset)
# pprint(data_process.tokenizer.decode(sample_dataset["ids"]))