from classifier import ClassifierBinary
from modelEncoderDecoderAdvancedV2 import IMEO
from itertools import product
import matplotlib.pyplot as plt
import torch
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utilsData import dataset_loader, load_data

# CLASSIFIER PARAMETERS
CL_batch_size = 100
CL_learning_rate = 0.002
CL_plot = False
CL_weight_decay = 0.8e-5
CL_num_epochs = 20

# ENCODER PARAMETERS
EN_binary_loss_weight = 0.5
EN_batch_size = 100
EN_learning_rate = 0.002
EN_plot = True
EN_embedding_dim_list = [17]
EN_weight_decay = 0.2e-5
EN_num_epochs = 300
EN_masked_percentage_list = [0.2]

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

for embedding_dim, masked_percentage in product(EN_embedding_dim_list, EN_masked_percentage_list):
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
    )
    encoder.saveModel(f'./Encoder_classifier/Models/encoder_{embedding_dim}_{masked_percentage}.pth')
    # we freeze the encoder
    encoder.freeze()
    # create classifier
    classifier = ClassifierBinary(inputSize=embedding_dim)
    # fit classifier
    classifier.fit(
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
    )
    
    classifier.saveModel(f'./Encoder_classifier/Models/classifier_{embedding_dim}_{masked_percentage}.pth')
    
    