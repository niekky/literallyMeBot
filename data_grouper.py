import json
import numpy as np
from pprint import pprint

class DataGrouper():
    def __init__(self):
        new_dataset = self.group()
        self.save(new_dataset)

    def group(self):
        user_input = ""
        output = []
        counter = 1
        while True:
            user_input = input(f"Enter filename {counter}: ")
            if user_input == "quit":
                break
            with open(f"data/{user_input}.json", "r") as f:
                this_dict = json.load(f)
                output = output + this_dict
            counter += 1
        return output
    
    def save(self, dict):
        user_input = input("New filename: ")
        if user_input != "quit":    
            with open(f"data/{user_input}.json", "w") as f:
                json.dump(dict, f)
            pprint("SAVE COMPLETED!")

DataGrouper()