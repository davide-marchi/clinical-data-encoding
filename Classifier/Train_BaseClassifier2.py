#only because i don't wnat ot break the other main.py
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning, append=True)
print("WARNING: 'ignoring future warnings' not working")

from time import time
from BaseClassifier import BaseClassifier
from itertools import product
import torch
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utilsData import dataset_loader_full, set_gpu, set_cpu, load_past_results_and_models
import json
from sklearn.metrics import classification_report, f1_score
from skorch import NeuralNetClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV


# YEARS TO PREDICT
years_to_death = 8
# CLASSIFIER PARAMETERS
param_grid = {
        'optimizer__lr' : [0.0001, 0.005, 0.3],
        'optimizer__weight_decay' : [1e-7, 5e-4],
        'max_epochs' : [10, 500],
        'batch_size' : [10, 1000]
    }

device = set_cpu()

# LOAD DATASET
dict = dataset_loader_full(years=years_to_death)
tr_data = dict['tr_data']
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

begin = time()

model = NeuralNetClassifier(
    module = BaseClassifier,
    module__inputSize = tr_data.shape[1],
    optimizer = torch.optim.Adam,
    device = device,
    criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([posWeight])),
    verbose=0,
    callbacks='disable'
)

grid = HalvingGridSearchCV(estimator=model, 
                            param_grid=param_grid, 
                            n_jobs=-1, 
                            verbose=1,
                            cv=3,
                            scoring='f1_macro',
                            )
# Reshape tr_out to be 2D
formatted_tr_out = tr_out.reshape(-1, 1)

# Use formatted_tr_out in the fit method
grid_result = grid.fit(tr_data, formatted_tr_out)

y_pred = grid.predict(val_data)
report = classification_report(val_out, y_pred, output_dict=True)

with open('./Classifier/results.json', 'w') as f:
    json.dump(report, f, indent=4)

with open('./Classifier/parametes.json', 'w') as f:
    json.dump(grid.best_params_, f, indent=4)
    
    
end = time()
print(f'using device: {device}')
tot_time = end - begin
print(f'Total time: {tot_time//60}m {tot_time%60}s')
#print(results)

