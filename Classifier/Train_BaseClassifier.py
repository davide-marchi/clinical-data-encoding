import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

import torch
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utilsData import dataset_loader
from BaseClassifier import BaseClassifier
from Evaluation_Metrics import plot_accuracy, plot_loss, plot_c_matrix, report_scores


# Define hyperparameters
LEARNING_RATE = 0.00005
EPOCHS = 300
TRAIN_BATCH_SIZE = 75
VALID_BATCH_SIZE = 35
WEIGHT_DECAY=0
GAMMA=0

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
    'shuffle':True,  
    'num_workers': 0  
}

# Create DataLoader for train and validation sets
train_dataset = TensorDataset(tr_data, tr_out)
train_loader = DataLoader(train_dataset, **train_params)
val_dataset = TensorDataset(val_data, val_out)
val_loader = DataLoader(val_dataset, **test_params)
test_dataset=TensorDataset(test_data, test_out)
test_loader = DataLoader(test_dataset, **test_params)

# Initialize the model
model = BaseClassifier(tr_data.shape[1])
model.to(device)

# Define optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=GAMMA)

# Train the model
train_loss, val_loss, train_accuracy, val_accuracy = model.fit_model(train_loader, val_loader, optimizer, scheduler, device, EPOCHS)

# Plot training loss and accuracy
plot_loss(train_losses=train_loss, val_losses=val_loss)
plot_accuracy(train_accuracies=train_accuracy, val_accuracies=val_accuracy)

test_prediction, test_target = model.test_model(test_loader, device)

report_scores(test_target, test_prediction)
plot_c_matrix(test_target, test_prediction, "Base Classifier")