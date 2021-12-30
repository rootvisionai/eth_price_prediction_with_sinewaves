# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 16:39:01 2021

@author: tekin.evrim.ozmermer
"""

import pandas as pd
import torch, torchvision



def import_data(filepath):
    df = pd.read_csv(filepath)
    return df

def prepare_data(df, column = "Close", training_data_ratio = 0.99):
    data_tensor = torch.from_numpy(df[column].values).unsqueeze(1)
    data_tensor = data_tensor[-4096:]
    data_tensor = data_tensor-data_tensor.min()
    data_tensor = data_tensor/data_tensor.max()
    training_data_length = int(data_tensor.shape[0]*training_data_ratio)
    train_tensor = data_tensor[0:training_data_length]
    # val_tensor = data_tensor[training_data_length:]
    return train_tensor, data_tensor
