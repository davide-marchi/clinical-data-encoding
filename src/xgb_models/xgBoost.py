import os
import sys
from random import shuffle
import json
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

from xgboost import XGBClassifier # XGBoost classifier
from time import time # to compute time
from itertools import product #for grid search
from tqdm import tqdm # for progress bar
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score # for evaluation
from utilsData import dataset_loader, load_data # for loading data
from imblearn.over_sampling import RandomOverSampler # for oversampling
f1_macro = lambda x, y: f1_score(x, y, average='macro') # for our evaluation

# YEARS TO PREDICT
years_to_death = 8
                                                           # _ O _
# Load data with dataset_loader to avoid segmentation fault   \|/
folderName = f'./Datasets/Cleaned_Dataset_{years_to_death}Y/chl_dataset_known.csv'
dataset = load_data(folderName)
dict_ = dataset_loader(dataset, 0.1, 0.2, 42)
tr_data, val_data, tr_out, val_out = dict_['tr_data'], dict_['val_data'], dict_['tr_out'], dict_['val_out']
ts_data, ts_out = dict_['test_data'], dict_['test_out']

# TODO: complete gridsearch adding various sampling strategies
oversample = RandomOverSampler(sampling_strategy='minority', random_state=42)
tr_data, tr_out = oversample.fit_resample(tr_data, tr_out)

# Hyperparameters
N_estimators = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400]
Learning_rate = [0.001, 0.005, 0.01, 0.05, 0.1]
# Shuffle hyperparameters
hyperparameters = list(product(N_estimators, Learning_rate))
shuffle(hyperparameters)

# Grid search
best_score = 0
best_params = None
for n_estimators, learning_rate in tqdm(hyperparameters, total=len(hyperparameters)):
    xgb_model = XGBClassifier(
        n_estimators=n_estimators, 
        learning_rate=learning_rate, 
        n_jobs=-1, 
        random_state=42,
        objective='binary:logistic'
    )
    xgb_model.fit(tr_data, tr_out)
    val_pred = xgb_model.predict(val_data)
    score = balanced_accuracy_score(val_out, val_pred)
    if score > best_score:
        best_score = score
        best_params = (n_estimators, learning_rate)

# Train the model with the best hyperparameters
n_estimators, learning_rate = best_params
xgb_model = XGBClassifier(
    n_estimators=n_estimators, 
    learning_rate=learning_rate, 
    n_jobs=-1, 
    random_state=42,
    objective='binary:logistic'
)
xgb_model.fit(tr_data, tr_out)

print(f'Best hyperparameters: {best_params}\n')

# Predict on validation data
val_pred = xgb_model.predict(val_data)
print('\033[1m\033[92m' + 'VALIDATION' + '\033[0m')
print(classification_report(val_out, val_pred))
results = [{
    'n_estimators': n_estimators,
    'learning_rate': learning_rate,
    'encoder': 'encoder_None_None_None_None_None_None_None_None',
    'results': classification_report(val_out, val_pred, output_dict=True)
}]

with open(f'./src/xgb_models/results_C_V_{years_to_death}Y.json', 'w') as f:
    json.dump(results, f, indent=4)

# Predict on test data
ts_pred = xgb_model.predict(ts_data)
print('\033[1m\033[92m' + 'TEST' + '\033[0m')
print(classification_report(ts_out, ts_pred))
results = [{
    'n_estimators': n_estimators,
    'learning_rate': learning_rate,
    'encoder': 'encoder_None_None_None_None_None_None_None_None',
    'results': classification_report(ts_out, ts_pred, output_dict=True)
}]

with open(f'./src/xgb_models/results_C_T_{years_to_death}Y.json', 'w') as f:
    json.dump(results, f, indent=4)
