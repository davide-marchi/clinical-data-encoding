import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
sys.path.insert(1, os.path.join(sys.path[0], '../../Encoder_classifier'))

# for the models
from utilsData import load_past_results_and_models # for loading trained models
from modelEncoderDecoderAdvancedV2 import IMEO # for loading the MIEO model
from utilsData import unpack_encoder_name # for unpacking the encoder name
import torch # for loading the models
from xgboost import XGBClassifier # XGBoost classifier
# for grid search
from itertools import product #for grid search
from tqdm import tqdm # for progress bar
# for evaluation
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score # for evaluation
f1_macro = lambda x, y: f1_score(x, y, average='macro') # for our evaluation
import json
# for loading data
from utilsData import dataset_loader, load_data # for loading data
from imblearn.over_sampling import RandomOverSampler # for oversampling
from random import shuffle
import numpy as np
from typing import Tuple

# YEARS TO PREDICT
years_to_death = 8
                                                           # _ O _
# Load data with dataset_loader to avoid segmentation fault   \|/
folderName = f'./Datasets/Cleaned_Dataset_{years_to_death}Y/chl_dataset_known.csv'
dataset = load_data(folderName)
dict_ = dataset_loader(dataset, 0.1, 0.2, 42)
tr_data, val_data, tr_out, val_out = dict_['tr_data'], dict_['val_data'], dict_['tr_out'], dict_['val_out']
binary_clumns = dict_['bin_col']

oversample = RandomOverSampler(sampling_strategy='minority', random_state=42)
tr_data, tr_out = oversample.fit_resample(tr_data, tr_out)
tr_data = torch.tensor(tr_data, dtype=torch.float32)

# Hyperparameters
_, models_names, _= load_past_results_and_models(years_to_death)
#models_names = models_names[1:2]
N_estimators = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400]
Learning_rate = [0.001, 0.005, 0.01, 0.05, 0.1]
# Shuffle hyperparameters
hyperparameters = list(product(N_estimators, Learning_rate, models_names))
shuffle(hyperparameters)

def get_model(model_name:str)->IMEO:
    return (torch.load(f'./Encoder_classifier/gridResults/Models/{model_name}', weights_only=False))

def encode_with_model(model_name:str) -> Tuple[np.array, np.array]:
    model = get_model(model_name)
    tr_data_enc = model.encode(tr_data).detach().numpy()
    val_data_enc = model.encode(val_data).detach().numpy()
    return tr_data_enc, val_data_enc

# Grid search
best_score = 0
best_params = None
results = []
altready_computed = []
with open(f'./src/xgb_models/results{years_to_death}_Y.txt', 'r') as f:
    results = json.load(f)

for elem in results:
    altready_computed.append((elem['n_estimators'], elem['learning_rate'], elem['encoder']))
    if elem['results']['macro avg']['f1-score'] > best_score:
        best_score = elem['results']['macro avg']['f1-score']
        best_params = (elem['n_estimators'], elem['learning_rate'], elem['encoder'])

print(f'Already computed: {len(altready_computed)}')
print(f'Best score: {best_score}')

for n_estimators, learning_rate, encoder in tqdm(hyperparameters, total=len(hyperparameters)):
    if (n_estimators, learning_rate, encoder) in altready_computed:
        continue
    tr_data_enc, val_data_enc = encode_with_model(encoder)
    xgb_model = XGBClassifier(
        n_estimators=n_estimators, 
        learning_rate=learning_rate, 
        n_jobs=2, 
        random_state=42,
        objective='binary:logistic'
    )
    xgb_model.fit(tr_data_enc, tr_out)
    val_pred = xgb_model.predict(val_data_enc)
    score = balanced_accuracy_score(val_out, val_pred)
    results.append({
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'encoder': encoder,
            'results': classification_report(val_out, val_pred, output_dict=True)
        })
    with open(f'./src/xgb_models/results{years_to_death}_Y.txt', 'w') as f:
            json.dump(results, f, indent=4)
            altready_computed.append((n_estimators, learning_rate, encoder))
    
    if score > best_score:
        best_score = score
        best_params = (n_estimators, learning_rate, encoder)

# Train the model with the best hyperparameters
n_estimators, learning_rate, model = best_params
xgb_model = XGBClassifier(
    n_estimators=n_estimators, 
    learning_rate=learning_rate, 
    n_jobs=-1, 
    random_state=42,
    objective='binary:logistic'
)
tr_data_enc, val_data_enc = encode_with_model(model)
xgb_model.fit(tr_data_enc, tr_out)

# Predict
val_pred = xgb_model.predict(val_data_enc)
print(f'val balanced acc: {balanced_accuracy_score(val_out, val_pred)}')
print(classification_report(val_out, val_pred))
print(f'Best hyperparameters: {best_params}')