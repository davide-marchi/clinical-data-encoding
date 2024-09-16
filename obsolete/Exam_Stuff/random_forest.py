from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
from modelEncoderDecoderAdvancedV2 import IMEO
import torch

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utilsData import dataset_loader, load_data
from sklearn.metrics import classification_report

#load the data

folderName = './Datasets/Cleaned_Dataset/'
fileName = 'chl_dataset.csv'
dataset = load_data(folderName + fileName)

dict = dataset_loader(dataset, 0.1, 0.2, 42)
tr_data = dict['tr_data']
tr_out = dict['tr_out']
val_data = dict['val_data']
val_out = dict['val_out']

encoder_decoder:IMEO = torch.load('./Encoder_classifier/encoder_decoder.pth')
tr_data = encoder_decoder.encode(tr_data).detach().numpy()
val_data = encoder_decoder.encode(val_data).detach().numpy()

print(f'Training data dimension: {tr_data.shape}')
#grid search for random forest
param_dist = {"max_depth": [2,3,5,6,7,10,12,None],
              "max_features": [None, "sqrt", "log2"] + list(range(1, len(val_data[0]) + 1)),
              "min_samples_split": randint(10, 51),
              "min_samples_leaf": randint(10, 51),
              "bootstrap": [True, False],
              "criterion": ["entropy", "gini"],
              "class_weight":['balanced', 'balanced_subsample', None]}
n_iter_search = 50
clf = RandomForestClassifier(n_estimators=30)
grid_search = RandomizedSearchCV(clf, param_distributions=param_dist, 
                            n_iter=n_iter_search, 
                            n_jobs=10, 
                            scoring=lambda clf, x, y: f1_score(y, clf.predict(x), average='macro'),
                            cv=5, verbose=2
                            )
grid_search.fit(tr_data, tr_out)

print('Best setting parameters\n', grid_search.cv_results_['params'][0])
print('\nMean of this setting\n', grid_search.cv_results_['mean_test_score'][0], 
      '\n\nStandard Deviation (std) of this setting\n', grid_search.cv_results_['std_test_score'][0])

#run the model
rf = RandomForestClassifier(n_estimators=30,
                             criterion='gini',
                             max_features='log2', 
                             min_samples_split=42,
                             min_samples_leaf=12,
                             max_depth = None,
                             bootstrap=True)
rf = rf.fit(tr_data, tr_out)
train_pred_rf = rf.predict(tr_data)

val_pred_rf = rf.predict(val_data)

print(classification_report(tr_out, train_pred_rf))
print(classification_report(val_out, val_pred_rf))