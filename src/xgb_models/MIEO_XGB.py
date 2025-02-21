import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
sys.path.insert(1, os.path.join(sys.path[0], '../../Encoder_classifier'))

# for the models
from utilsData import load_past_results_and_models # for loading trained models
from src.xgb_models.utility_xgb import encode_with_model, get_dataset # for encoding the data
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
from sklearn.model_selection import cross_val_score

# YEARS TO PREDICT
years_to_death = 8
dict_ = get_dataset()
tr_data, val_data, tr_out, val_out = dict_['tr_data'], dict_['val_data'], dict_['tr_out'], dict_['val_out']

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
    tr_data_enc, val_data_enc = encode_with_model(encoder, tr_data, val_data)
    xgb_model = XGBClassifier(
        n_estimators=n_estimators, 
        learning_rate=learning_rate, 
        n_jobs=-1, 
        random_state=42,
        objective='binary:logistic'
    )
    # Perform cross-validation
    scores = cross_val_score(xgb_model, tr_data_enc, tr_out, cv=4, scoring='balanced_accuracy', n_jobs=-1)
    score = scores.mean()
    results.append({
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'encoder': encoder,
            'result': score
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
tr_data_enc, val_data_enc = encode_with_model(model, tr_data, val_data)
xgb_model.fit(tr_data_enc, tr_out)

# Predict
val_pred = xgb_model.predict(val_data_enc)
print(f'val balanced acc: {balanced_accuracy_score(val_out, val_pred)}')
print(classification_report(val_out, val_pred))
print(f'Best hyperparameters: {best_params}')