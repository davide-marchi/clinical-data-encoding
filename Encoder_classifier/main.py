#only because i don't wnat ot break the other main.py
import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)

from time import time
from classifier import ClassifierBinary
from modelEncoderDecoderAdvancedV2 import IMEO
from itertools import product
from tqdm import tqdm
import torch
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utilsData import dataset_loader_full, set_gpu, set_cpu, load_past_results_and_models, is_intel_xeon
import json
from sklearn.metrics import classification_report, f1_score
from skorch import NeuralNetClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV
from skorch.callbacks import EarlyStopping

xeonFlag = is_intel_xeon()
if xeonFlag:
    import intel_extension_for_pytorch as ipex

# YEARS TO PREDICT
years_to_death = 8
# CLASSIFIER PARAMETERS
param_grid = {
        'optimizer__lr' : [0.0005, 0.001, 0.005, 0.01],
        'optimizer__weight_decay' : [0.01e-5, 0.05e-5, 0.1e-5, 0.5e-5],
        'max_epochs' : [50],
        'batch_size' : [100, 300]
    }
# ENCODER PARAMETERS
EN_binary_loss_weight = [ 0.001, 0.01, 0.5, None]# very important
EN_batch_size = [300]
EN_learning_rate = [0.0015, 0.003]
EN_plot = False
EN_embedding_perc_list = [2, 3, 0.8, 0.5]
EN_weight_decay = [0.05e-5, 0.25e-5, 0.5e-5]
EN_num_epochs = [250]
EN_masked_percentage_list = [ 0.2, 0.35, 0.5]
EN_patience = [10]

device = set_cpu()

# LOAD DATASET
dict = dataset_loader_full(years=years_to_death)
tr_data = dict['tr_data']
extended_tr_data = dict['tr_unlabled']
val_data = dict['val_data']
tr_out = dict['tr_out']
val_out = dict['val_out']
binary_clumns = dict['bin_col']
#count the pos number
posCount = tr_out.sum()
negCount = tr_out.shape[0] - posCount
posWeight = negCount/posCount
print(f'Positive count: {posCount}')
print(f'Negative count: {negCount}')
print(f'Positive weight: {posWeight}')
print(f'Shape of tr_out: {tr_out.shape}')

# json      models in directory     validated models
results,    existing_models,        validated_models = load_past_results_and_models()

begin = time()
combinations = list(product(EN_binary_loss_weight, EN_batch_size, EN_learning_rate, EN_embedding_perc_list, EN_weight_decay, EN_num_epochs, EN_masked_percentage_list, EN_patience))
combinations.insert(0, (None,)*len(combinations[0]))
print(f'Number of combinations: {len(combinations)}')
for comb in tqdm(combinations, desc="Processing combinations", colour="green"):
    en_bin_loss_w, en_bs, en_lr, en_emb_perc, en_wd, en_num_ep, en_masked_perc, en_pt = comb
    torch.manual_seed(42)

    encoder_string = f'encoder_{en_bin_loss_w}_{en_bs}_{en_lr}_{en_emb_perc}_{en_wd}_{en_num_ep}_{en_masked_perc}_{en_pt}'

    ################################################################################################
    # TRAIN ENCODER ################################################################################
    ################################################################################################
    if comb != (None,)*len(comb):
        if encoder_string in validated_models:
            print(f'{encoder_string} already validated')
            continue
        # create, train and save encoder
        encoder = IMEO(
            inputSize=tr_data.shape[1], 
            total_binary_columns=binary_clumns, 
            embedding_percentage=en_emb_perc,
            )
        # check if encoder exists
        if encoder_string + '.pth' in existing_models:
            # load encoder
            encoder:IMEO = torch.load('./Encoder_classifier/gridResults/Models/' + encoder_string + '.pth', weights_only=False)
            #encoder.load_state_dict(torch.load('./Encoder_classifier/gridResults/Models/' + encoder_string + '.pth', weights_only=True))
        else:
            optimizer = torch.optim.Adam(encoder.parameters(),weight_decay=en_wd, lr=en_lr)
            if xeonFlag:
                encoder, optimizer = ipex.optimize(encoder, optimizer=optimizer)
            encoder.fit(
                extended_tr_data, 
                val_data,
                optimizer = optimizer,
                device = device,
                batch_size = en_bs,
                plot = EN_plot,
                num_epochs = en_num_ep,
                masked_percentage = en_masked_perc,
                early_stopping=en_pt,
                binary_loss_weight=en_bin_loss_w,
                print_every=en_num_ep,
            )
            encoder.saveModel(f'./Encoder_classifier/gridResults/Models/{encoder_string}.pth')
            existing_models.append(encoder_string + '.pth')

        encoder.freeze()
        encoded_tr_data = encoder.encode(tr_data)
        val_data_encoded = encoder.encode(val_data)
        embedding_dim = encoder.embedding_dim
    else:
        encoded_tr_data = tr_data.clone().detach()
        val_data_encoded = val_data.clone().detach()
        embedding_dim = tr_data.shape[1]

    ################################################################################################
    # TRAIN CLASSIFIER #############################################################################
    ################################################################################################


    model = NeuralNetClassifier(
        module = ClassifierBinary,
        module__inputSize = embedding_dim,
        optimizer = torch.optim.Adam,
        device = device,
        criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([posWeight])),#TODO: this will be a hyperparameter (if they pay us enough)
        verbose=0,
        callbacks=[('early_stopping', EarlyStopping(patience=10))]
    )

    grid = GridSearchCV(estimator=model, 
                               param_grid=param_grid, 
                               n_jobs=-1,
                               verbose=0,
                               scoring='balanced_accuracy',
                               cv=4,
                               #random_state=42, //For halving search
                               )
    
    grid_result = grid.fit(encoded_tr_data, tr_out)
    y_pred = grid.predict(val_data_encoded)
    report = classification_report(val_out, y_pred, output_dict=True)

    results.append({
        'encoder_string': encoder_string,
        'encoder': {
            'binary_loss_weight': en_bin_loss_w,
            'batch_size': en_bs,
            'lr': en_lr,
            'emb_perc': en_emb_perc,
            'wd': en_wd,
            'num_ep': en_num_ep,
            'masked_perc': en_masked_perc,
            'pt': en_pt
        },
        'classifier': grid.best_params_,
        'results': report
    })
    with open('./Encoder_classifier/gridResults/results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    
end = time()
print(f'using device: {device}')
tot_time = end - begin
print(f'Total time: {tot_time//60}m {tot_time%60}s')
#print(results)

