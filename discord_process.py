import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
from pprint import pprint
import json
import emoji
import re

class DiscordProcess():
    def __init__(self):
        self.raw = pd.read_csv("data\DM_Kaiyzxn.csv")
        self.data_length = self.raw.shape[0]
        self.raw = self.process_csv(self.raw)
        data_dict = self.sentence_group(self.raw)
        self.save_csv2json(data_dict, "DM_Kaiyzxn")
    
    def process_csv(self, dataframe):
        data_length = dataframe.shape[0]
        url_regex = r'[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
        filter_char = lambda c: ord(c) < 256
        dataframe = dataframe.drop(dataframe[~ dataframe.Attachments.isnull()].index)
        dataframe = dataframe.drop(dataframe[dataframe.Content.str.contains(url_regex, regex = True)].index)
        dataframe = dataframe.drop(dataframe[dataframe.Content.str.contains("Started a call that lasted")].index)
        
        dataframe.Content = dataframe.Content.apply(lambda s: ''.join(filter(filter_char, s)))
        dataframe.Content = dataframe.Content.str.replace('\"','')
        dataframe.Content = dataframe.Content.str.replace('\'','')
        dataframe.Content = dataframe.Content.str.replace(".",'')
        dataframe.Content = dataframe.Content.str.replace("[",'')
        dataframe.Content = dataframe.Content.str.replace("]",'')
        dataframe.Content = dataframe.Content.str.replace("{",'')
        dataframe.Content = dataframe.Content.str.replace("}",'')
        dataframe.Content = dataframe.Content.str.replace("(",'')
        dataframe.Content = dataframe.Content.str.replace(")",'')
        dataframe.Content = dataframe.Content.str.replace("\n","")
        dataframe = dataframe.drop(dataframe[dataframe.Content == ""].index)
        dataframe.Content = dataframe.Content.str.strip()
        return dataframe

    def sentence_conv1_1(self, raw_dataset):
        length = raw_dataset.shape[0]
        dialog = []
        output = raw_dataset.iloc[0]["Content"]
        for i in range(length):
            curr = raw_dataset.iloc[i]
            # print(curr["Content"])
            next = raw_dataset.iloc[i+1 if (i+1) < length else i]
            if curr["Author"] != next["Author"]:
                if output.strip()!="":
                    dialog.append(
                        {curr["Author"]: output.strip() if output[0]!=',' else output[1:].strip()}
                        )
                output = ""
            output = output + " " + str(next["Content"])
        return dialog
    
    def sentence_group(self, raw_dataset):
        length = raw_dataset.shape[0]
        dialog = []
        target_bot = ".nick#2667"
        output = raw_dataset.iloc[0]["Content"]
        for i in range(length):
            curr = raw_dataset.iloc[i]
            next = raw_dataset.iloc[i+1 if (i+1) < length else i]
            curr["Author"] = "other" if curr["Author"] != target_bot else curr["Author"]
            next["Author"] = "other" if next["Author"] != target_bot else next["Author"]

            if curr["Author"] != next["Author"]:
                if output.strip()!="":
                    dialog.append(
                        {curr["Author"]: output.strip() if output[0]!=',' else output[1:].strip()}
                        )
                    output = ""
            output = output + " " + str(next["Content"])
        return dialog
    
    def save_csv2json(self, data_dict, name):
        with open(f"data\{name}.json", "w") as f :
            json.dump(data_dict, f)
        print("SAVED!")
    
    def load_json(self, name):
        with open(f"data\{name}.json", "r") as f:
            data_dict = json.load(f)
        return data_dict

DiscordProcess()