import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
import json
from nlpaug.util import Action
from data_process import DataProcess
from pprint import pprint


def augment_data(pairs):
    aug_pairs = []

    spelling_aug = naw.SpellingAug()

    context_aug_sub = naw.ContextualWordEmbsAug(
        model_path = "bert-base-uncased",
        action = "substitute"
    )

    context_aug_ins = naw.ContextualWordEmbsAug(
        model_path = "bert-base-uncased",
        action = "insert"
    )

    syn_aug = naw.SynonymAug(aug_src = "wordnet")

    for idx, pair in enumerate(pairs):
        sentence = pair[0]
        target = pair[1]
        
        cont_aug_text_sub = [[context_aug_sub.augment(sentence, n=1)[0], target]]
        cont_aug_text_sub2 = [[context_aug_sub.augment(sentence, n=1)[0], target]]
        cont_aug_text_ins = [[context_aug_ins.augment(sentence, n=1)[0], target]]
        cont_aug_text_ins2 = [[context_aug_ins.augment(sentence, n=1)[0], target]]
        spell_aug_text = [[spelling_aug.augment(sentence, n=1)[0], target]]
        spell_aug_text2 = [[spelling_aug.augment(sentence, n=1)[0], target]]
        syn_aug_text = [[syn_aug.augment(sentence, n = 1)[0], target]]
        syn_aug_text2 = [[syn_aug.augment(sentence, n = 1)[0], target]]
        aug_set = cont_aug_text_sub + cont_aug_text_sub2 + cont_aug_text_ins + cont_aug_text_ins2 + spell_aug_text + spell_aug_text2 + syn_aug_text + syn_aug_text2

        aug_pairs = aug_pairs + aug_set

    return aug_pairs        

data_process = DataProcess("friends_conv")
sample = data_process.get_pairs()

aug_data = augment_data(sample)
# pprint(aug_data)
print(len(aug_data))

with open("data/aug_pairs2.json", "w") as f:
    json.dump(aug_data, f)

# print("SRC: " + sentence)
# print(aug_sentence)
# print(aug_sentence2)
# print(spelling_aug)
# print(syn)


