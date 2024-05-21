from classifier import ClassifierBinary
from modelEncoderDecoderAdvancedV2 import IMEO
from itertools import product
import matplotlib.pyplot as plt
import torch
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utilsData import dataset_loader, load_data
import json

# TODO: for plotting the 3D plot !!!!!see the link below!!!!subscribe to the channel
# https://matplotlib.org/stable/gallery/mplot3d/contourf3d_2.html#sphx-glr-gallery-mplot3d-contourf3d-2-py

# CLASSIFIER PARAMETERS
CL_batch_size = 200
CL_learning_rate = 0.001
CL_plot = False
CL_weight_decay = 0.6e-5
CL_num_epochs = 20
CL_patience = 10

# ENCODER PARAMETERS
EN_binary_loss_weight = 0.6
EN_batch_size = 200
EN_learning_rate = 0.0015
EN_plot = False
EN_embedding_dim_list = [10, 20, 30]
EN_weight_decay = 0.05e-5
EN_num_epochs = 20
EN_masked_percentage_list = [0.1, 0.2, 0.3]
EN_patience = 20

device = torch.device(  "cuda" if torch.cuda.is_available() 
                        else  "mps" if torch.backends.mps.is_available()
                        else "cpu"
                    )
#we have small amount of data, so we will use cpu (looks faster)
device = torch.device("cpu")
print("Device: ", device)

folderName = './Datasets/Cleaned_Dataset/'
fileName = 'chl_dataset.csv'
dataset = load_data(folderName + fileName)

dict = dataset_loader(dataset, 0.1, 0.2, 42)
tr_data = dict['tr_data']
tr_out = dict['tr_out']
val_data = dict['val_data']
val_out = dict['val_out']
binary_clumns = dict['bin_col']

results = []
    
for embedding_dim, masked_percentage in product(EN_embedding_dim_list, EN_masked_percentage_list):
    print(f'\n\nEmbedding dim: {embedding_dim}, Masked percentage: {masked_percentage}\n')
    # create encoder
    encoder = IMEO(
        inputSize=tr_data.shape[1], 
        total_binary_columns=binary_clumns, 
        embedding_dim=embedding_dim,
        neurons_num=[100, 80, 40]
        )
    # fit encoder
    encoder.fit(
        tr_data, 
        val_data,
        optimizer = torch.optim.Adam(encoder.parameters(), weight_decay=EN_weight_decay, lr=EN_learning_rate),
        device = device,
        binary_loss_weight = EN_binary_loss_weight, 
        batch_size = EN_batch_size,
        plot = EN_plot,
        num_epochs = EN_num_epochs, 
        masked_percentage = masked_percentage,
        early_stopping=EN_patience
    )
    encoder.saveModel(f'./Encoder_classifier/Models/encoder_{embedding_dim}_{masked_percentage}.pth')
    # we freeze the encoder
    encoder.freeze()
    # create classifier
    classifier = ClassifierBinary(inputSize=embedding_dim)
    # fit classifier
    trLoss, vlLoss, trAcc, vlAcc = classifier.fit(
        tr_data, 
        tr_out, 
        val_data, 
        val_out, 
        optimizer=torch.optim.Adam(classifier.parameters(), lr=CL_learning_rate), 
        device=device, 
        num_epochs=CL_num_epochs, 
        batch_size=CL_batch_size, 
        preprocess=encoder.encode,
        print_every=CL_num_epochs//10,
        early_stopping=CL_patience,
        plot=CL_plot
    )
    
    classifier.saveModel(f'./Encoder_classifier/Models/classifier_{embedding_dim}_{masked_percentage}.pth')
    
    results.append((embedding_dim, masked_percentage, trAcc, vlAcc, trLoss, vlLoss))

print(results)

with open('./Encoder_classifier/Models/results.json', 'w') as f:
    json.dump(results, f)