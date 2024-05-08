import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import torch
import torch.nn as nn
import torch.utils
from torch.utils.data import DataLoader, Dataset

from utilsData import dataset_loader

script_directory = os.getcwd()

print(script_directory)
folder_name = 'Datasets/Cleaned_Dataset'
folder_path = os.path.join(script_directory, folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
data=pd.read_csv(f'{folder_path}/chl_dataset.csv')


dict=dataset_loader(data, 0.1, 0.2, 42)

tr_data=dict["tr_data"]

print(tr_data.shape)