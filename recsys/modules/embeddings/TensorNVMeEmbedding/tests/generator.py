import numpy as np
import os

def generate_weight(row, embed_num):
    weight = np.random.randn(row,embed_num)
    return weight

def generate_freq(row):
    freq = np.random.randint(0,row*10,row)
    return freq

weight = generate_weight(100,20)
folder_path = "weight_folder"
np.save(os.path.join(folder_path,"001"),weight)