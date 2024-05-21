import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import itertools

import torch
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utilsData import dataset_loader
from BaseClassifier import BaseClassifier

# Definisci gli iperparametri da testare
LEARNING_RATES = [1e-04]
EPOCHS = [200]
TRAIN_BATCH_SIZES = [50]
VALID_BATCH_SIZES = [25]
WEIGHT_DECAYS = [1e-6]
GAMMA = [0, 0.1]

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

# Lista per memorizzare i risultati della ricerca a griglia
results = []

# Itera su tutte le combinazioni di iperparametri
for lr, epochs, train_batch_size, valid_batch_size, weight_decay, gamma in itertools.product(LEARNING_RATES, EPOCHS, TRAIN_BATCH_SIZES, VALID_BATCH_SIZES, WEIGHT_DECAYS, GAMMA):
    
    # Definisci il DataLoader per il training set
    train_dataset = TensorDataset(tr_data, tr_out)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=0)
    
    # Definisci il DataLoader per il validation set
    val_dataset = TensorDataset(val_data, val_out)
    val_loader = DataLoader(val_dataset, batch_size=valid_batch_size, shuffle=True, num_workers=0)
    
    # Definisci il modello
    model = BaseClassifier(tr_data.shape[1])  # Assicurati di sostituire input_size con il valore corretto
    
    # Definisci ottimizzatore e scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=gamma)  # Esempio di scheduler
    
    # Addestra il modello
    train_loss, val_loss, train_accuracy, val_accuracy = model.fit_model(train_loader, val_loader, optimizer, scheduler, device, epochs)
    
    # Calcola la media della val_loss finale
    final_val_loss = val_loss[-1]
    final_val_accuracy = val_accuracy[-1]
    
    # Memorizza i risultati della ricerca a griglia
    results.append({
        'lr': lr,
        'epochs': epochs,
        'train_batch_size': train_batch_size,
        'valid_batch_size': valid_batch_size,
        'weight_decay': weight_decay,
        'gamma': gamma,
        'final_val_loss': final_val_loss,
        'final_val_accuracy': final_val_accuracy
    })

# Trova la configurazione migliore
best_config = max(results, key=lambda x: x['final_val_loss'])

# Stampare la configurazione migliore
print("Best Configuration:")
print(best_config)
