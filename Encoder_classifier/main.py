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
from sklearn.metrics import classification_report

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
EN_binary_loss_weight = 0.6 # not used
EN_batch_size = 200
EN_learning_rate = 0.0015
EN_plot = False
EN_embedding_perc_list = [0.1, 0.5, 1]
EN_weight_decay = 0.05e-5
EN_num_epochs = 200
EN_masked_percentage_list = [0, 0.25, 0.5]
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
    
for embedding_perc, masked_percentage in product(EN_embedding_perc_list, EN_masked_percentage_list):
    print(f'\n\nEmbedding perc: {embedding_perc}, Masked percentage: {masked_percentage}\n')
    # create encoder
    encoder = IMEO(
        inputSize=tr_data.shape[1], 
        total_binary_columns=binary_clumns, 
        embedding_percentage=embedding_perc
        )
    # fit encoder
    encoder.fit(
        tr_data, 
        val_data,
        optimizer = torch.optim.Adam(encoder.parameters(), weight_decay=EN_weight_decay, lr=EN_learning_rate),
        device = device, 
        batch_size = EN_batch_size,
        plot = EN_plot,
        num_epochs = EN_num_epochs, 
        masked_percentage = masked_percentage,
        early_stopping=EN_patience
    )
    encoder.saveModel(f'./Encoder_classifier/Models/encoder_{embedding_perc}_{masked_percentage}.pth')
    # we freeze the encoder
    encoder.freeze()
    # create classifier
    classifier = ClassifierBinary(inputSize=encoder.embedding_dim)
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
    
    classifier.saveModel(f'./Encoder_classifier/Models/classifier_{embedding_perc}_{masked_percentage}.pth')

    y_pred = torch.round(classifier.forward(encoder.encode(val_data))).detach().numpy()

    report = classification_report(val_out, y_pred, output_dict=True)
    
    dict_params = {
        'embedding_dim': encoder.embedding_dim,
        'embedding_perc': embedding_perc,
        'masked_percentage': masked_percentage,
        'trAcc': trAcc,
        'vlAcc': vlAcc,
        'trLoss': trLoss,
        'vlLoss': vlLoss,
        'report': report
    }
    
    results.append(dict_params)
    
    with open('./Encoder_classifier/Models/results.json', 'w') as f:
        json.dump(results, f, indent=4)

print(results)

