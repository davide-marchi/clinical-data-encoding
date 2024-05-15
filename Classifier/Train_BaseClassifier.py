import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import torch
from torch.utils.data import DataLoader, TensorDataset

from utilsData import dataset_loader
from BaseClassifier import BaseClassifier


# Define hyperparameters
LEARNING_RATE = 1e-05
EPOCHS = 20
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4

# Define device (use "cpu" since the dataset is small)
device = torch.device("cpu")

# Get current script directory
script_directory = os.getcwd()

print(script_directory)

# Define folder paths for dataset
folder_name = 'Datasets/Cleaned_Dataset'
folder_path = os.path.join(script_directory, folder_name)

# Create folder if it doesn't exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Load dataset
data = pd.read_csv(f'{folder_path}/chl_dataset.csv')

# Split dataset into train, validation, and test sets
data_dict = dataset_loader(data, 0.1, 0.2, 42)

tr_data = data_dict["tr_data"]
tr_out = data_dict["tr_out"]
val_data = data_dict["val_data"]
val_out = data_dict["val_out"]
test_data = data_dict["test_data"]
test_out = data_dict["test_out"]

# Set random seed for reproducibility
torch.manual_seed(42)

# Define parameters for DataLoader
train_params = {
    'batch_size': TRAIN_BATCH_SIZE,  
    'shuffle': True,  
    'num_workers': 0  
}
test_params = {
    'batch_size': VALID_BATCH_SIZE,  
    'shuffle': True,  
    'num_workers': 0  
}

# Create DataLoader for train and validation sets
train_dataset = TensorDataset(tr_data, tr_out)
train_loader = DataLoader(train_dataset, **train_params)
val_loader = DataLoader((val_data, val_out), **test_params)

# Initialize the model
model = BaseClassifier(tr_data.shape[1])
model.to(device)

# Define optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

# Train the model
train_loss, train_accuracy = model.train_model(train_loader, optimizer, device, EPOCHS)

# Plot training loss and accuracy
model.plot_loss(train_loss)
model.plot_accuracy(train_accuracy)

