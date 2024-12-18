import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np # for numerical operations
from xgboost import XGBClassifier # XGBoost classifier
from time import time # to compute time
from itertools import product #for grid search
from tqdm import tqdm # for progress bar
from sklearn.metrics import classification_report, f1_score # for evaluation
from utilsData import dataset_loader_full # for loading data
from utilsData import load_past_results_and_models # for loading trained models
from imblearn.over_sampling import RandomOverSampler # for oversampling

# YEARS TO PREDICT
years_to_death = 8
dict_ = dataset_loader_full(years=years_to_death)
tr_data = dict_['tr_data']
extended_tr_data = dict_['tr_unlabled']
val_data = dict_['val_data']
tr_out = dict_['tr_out']
val_out = dict_['val_out']
binary_clumns = dict_['bin_col']

oversample = RandomOverSampler(sampling_strategy='minority')
tr_data, tr_out = oversample.fit_resample(tr_data, tr_out)

# in case we want to start from encoded data also
#_, existing_models, _ = load_past_results_and_models()
print("Creating XGBoost model")
xgb_model = XGBClassifier(
    n_estimators=40, 
    learning_rate=0.001, 
    n_jobs=1, 
    random_state=42,
    objective='binary:logistic'
)
print("Training XGBoost model")
begin = time()
xgb_model.fit(tr_data, tr_out)
print(f"Training time: {time()-begin}")

# Predictions
print("Predicting")
tr_pred = xgb_model.predict(tr_data)
val_pred = xgb_model.predict(val_data)

# Evaluation
print('Training set')
print(classification_report(tr_out, tr_pred))
print('Validation set')
print(classification_report(val_out, val_pred))

# Save model
print("Saving model")
#xgb_model.save_model('xgboost_models/xgb_model.json')