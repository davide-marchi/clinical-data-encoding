from time import time
from classifier import ClassifierBinary
from modelEncoderDecoderAdvancedV2 import IMEO
from itertools import product
import matplotlib.pyplot as plt
import torch
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utilsData import dataset_loader, load_data, unpack_encoder_name
import json
from sklearn.metrics import classification_report

# TODO: for plotting the 3D plot !!!!!see the link below!!!!subscribe to the channel
# https://matplotlib.org/stable/gallery/mplot3d/contourf3d_2.html#sphx-glr-gallery-mplot3d-contourf3d-2-py

# CLASSIFIER PARAMETERS
CL_batch_size = [200]
CL_learning_rate = [0.0001, 0.0002, 0.0004]
CL_plot = False
CL_weight_decay = [0.2e-5, 0.5e-5]
CL_num_epochs = [50]
CL_patience = [5]
CL_loss_weight = [(0.3, 0.7), (0.25, 0.75), (0.5, 0.5)]
# ENCODER PARAMETERS
EN_binary_loss_weight = [None, 0.5]
EN_batch_size = [200]
EN_learning_rate = [0.0015, 0.002]
EN_plot = False
EN_embedding_perc_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3]
EN_weight_decay = [0.05e-5, 0.2e-5]
EN_num_epochs = [250]
EN_masked_percentage_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
EN_patience = [10]

tot_models = len(CL_batch_size)*len(CL_learning_rate)*len(CL_weight_decay)*len(CL_num_epochs)*len(CL_patience)*len(CL_loss_weight)*\
    len(EN_binary_loss_weight)*len(EN_batch_size)*len(EN_learning_rate)*len(EN_embedding_perc_list)*len(EN_weight_decay)*len(EN_num_epochs)*len(EN_masked_percentage_list)*len(EN_patience)
print(f'Total number of models: {tot_models}')
tot_min = (5/9)*tot_models
print(f'Expected time: {tot_min//60}h {tot_min%60}m')

'''
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
'''

device = torch.device(  "cuda" if torch.cuda.is_available() 
                        else  "mps" if torch.backends.mps.is_available()
                        else "cpu"
                    )
#we have small amount of data, so we will use cpu (looks faster)
#device = torch.device("cpu")
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
existing_models = []
if os.path.exists('./Encoder_classifier/Models/results.json'):
    with open('./Encoder_classifier/Models/results.json', 'r') as f:
        results = json.load(f)
if os.path.exists('./Encoder_classifier/Models/'):
    for file in os.listdir('./Encoder_classifier/Models/'):
        if 'encoder' in file:
            existing_models.append(file)
begin = time()
for en_bin_loss_w, en_bs, en_lr, en_emb_perc, en_wd, en_num_ep, en_masked_perc, en_pt,\
    cl_bs, cl_lr, cl_wd, cl_num_ep, cl_pt, cl_loss_w in \
        product(EN_binary_loss_weight, EN_batch_size, EN_learning_rate, EN_embedding_perc_list, EN_weight_decay, EN_num_epochs, EN_masked_percentage_list, EN_patience,\
                CL_batch_size, CL_learning_rate, CL_weight_decay, CL_num_epochs, CL_patience, CL_loss_weight):
    encoder_string = f'encoder_{en_bin_loss_w}_{en_bs}_{en_lr}_{en_emb_perc}_{en_wd}_{en_num_ep}_{en_masked_perc}_{en_pt}'
    classifier_string = f'classifier_{cl_bs}_{cl_lr}_{cl_wd}_{cl_num_ep}_{cl_pt}_{cl_loss_w}'
    print( unpack_encoder_name(encoder_string))
    print(f'Classifier: {classifier_string}')

    #check if the model already exists
    if classifier_string + '_' + encoder_string + '.pth' in existing_models:
        print('Model already exists\n')
        continue
    ################################################################################################
    # TRAIN ENCODER ################################################################################
    ################################################################################################
    # check if encoder exists
    if encoder_string + '.pth' in existing_models:
        # load encoder
        encoder:IMEO = torch.load('./Encoder_classifier/Models/' + encoder_string + '.pth') 
        print('Encoder loaded')
    else:
        # create, train and save encoder
        encoder = IMEO(
            inputSize=tr_data.shape[1], 
            total_binary_columns=binary_clumns, 
            embedding_percentage=en_emb_perc,
            )
        encoder.fit(
            tr_data, 
            val_data,
            optimizer = torch.optim.Adam(encoder.parameters(), 
                                        weight_decay=en_wd, 
                                        lr=en_lr
                                        ),
            device = device,
            batch_size = en_bs,
            plot = EN_plot,
            num_epochs = en_num_ep,
            masked_percentage = en_masked_perc,
            early_stopping=en_pt,
            binary_loss_weight=en_bin_loss_w,
            print_every=en_num_ep,
        )
        encoder.saveModel(f'./Encoder_classifier/Models/{encoder_string}.pth')
        existing_models.append(encoder_string + '.pth')

    encoder.freeze()
    ################################################################################################
    # TRAIN CLASSIFIER #############################################################################
    ################################################################################################
    classifier = ClassifierBinary(inputSize=encoder.embedding_dim)
    trLoss, vlLoss, trAcc, vlAcc = classifier.fit(
        tr_data, tr_out, 
        val_data, val_out, 
        optimizer=torch.optim.Adam(classifier.parameters(), 
                                   lr=cl_lr, 
                                   weight_decay=cl_wd,
                                   ), 
        device=device, 
        num_epochs=cl_num_ep,
        batch_size=cl_bs,
        preprocess=encoder.encode,
        print_every=cl_num_ep,
        early_stopping=cl_pt,
        plot=CL_plot,
        loss_weight=cl_loss_w,
    )
    classifier.saveModel(f'./Encoder_classifier/Models/{classifier_string}_{encoder_string}.pth')

    ################################################################################################
    # EVALUATE #####################################################################################
    ################################################################################################
    val_data = val_data.to(device)
    y_pred = torch.round(classifier.forward(encoder.encode(val_data))).detach().cpu().numpy()
    report = classification_report(val_out, y_pred, output_dict=True)
    
    dict_params = {
        'encoder': encoder_string,
        'classifier': classifier_string,
        'embedding_dim': encoder.embedding_dim,
        'embedding_perc': en_emb_perc,
        'masked_percentage': en_masked_perc,
        'trAcc': trAcc,
        'vlAcc': vlAcc,
        'trLoss': trLoss,
        'vlLoss': vlLoss,
        'report': report
    }
    
    results.append(dict_params)
    
    with open('./Encoder_classifier/Models/results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
end = time()
print(f'using device: {device}')
tot_time = end - begin
print(f'Total time: {tot_time//60}m {tot_time%60}s')
#print(results)

