import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utilsData import dataset_loader_full, load_past_results_and_models, unpack_encoder_name
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from Encoder_classifier.classifier import ClassifierBinary
from skorch import NeuralNetClassifier
import json
import torch
from skorch.callbacks import EarlyStopping
from src.xgb_models.utility_xgb import encode_with_model

results, _, _ = load_past_results_and_models()

ENCODER_TEST = True
CLASSIFER_ONLY = True

years_to_death = 8

# LOAD DATASET
dict = dataset_loader_full(years=years_to_death)
tr_data = dict['tr_data']
extended_tr_data = dict['tr_unlabled']
test_data = dict['test_data']
tr_out = dict['tr_out']
test_out = dict['test_out']
binary_clumns = dict['bin_col']
#count the pos number
posCount = tr_out.sum()
negCount = tr_out.shape[0] - posCount
posWeight = negCount/posCount

if CLASSIFER_ONLY:
    
    hyperparameters=results[0]["classifier"]
    
    classifier= NeuralNetClassifier(
        module = ClassifierBinary,
        module__inputSize = tr_data.shape[1],
        optimizer = torch.optim.Adam,
        optimizer__lr = hyperparameters['optimizer__lr'],
        optimizer__weight_decay = hyperparameters['optimizer__weight_decay'],
        max_epochs = hyperparameters['max_epochs'],
        batch_size = hyperparameters['batch_size'],
        criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([posWeight])),#TODO: this will be a hyperparameter (if they pay us enough)
        verbose=0,
        callbacks=[('early_stopping', EarlyStopping(patience=10))]
    )
    
    classifier.fit(tr_data, tr_out)
    test_pred = classifier.predict(test_data)
    report_C = classification_report(test_out, test_pred, output_dict=False)
    print(report_C)
    cm = confusion_matrix(test_out, test_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1'])
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig('./Figures/Test_NN_C.png')
    plt.show()

if ENCODER_TEST:
    
    sorted_results = sorted(results, key=lambda x: x['results']['macro avg']['recall'], reverse=True)
    position = 0
    bestModel = sorted_results[position]['encoder_string']+".pth"
    bestClassifier = sorted_results[position]['classifier']
    tr_data_enc, test_data_enc = encode_with_model(bestModel, tr_data, test_data)
    classifier = NeuralNetClassifier(
        module = ClassifierBinary,
        module__inputSize = tr_data_enc.shape[1],
        optimizer = torch.optim.Adam,
        optimizer__lr = bestClassifier['optimizer__lr'],
        optimizer__weight_decay = bestClassifier['optimizer__weight_decay'],
        max_epochs = bestClassifier['max_epochs'],
        batch_size = bestClassifier['batch_size'],
        criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([posWeight])),#TODO: this will be a hyperparameter (if they pay us enough)
        verbose=0,
        callbacks=[('early_stopping', EarlyStopping(patience=10))]
    )
    classifier.fit(tr_data_enc, tr_out)
    test_pred = classifier.predict(test_data_enc)
    report_CE = classification_report(test_out, test_pred, output_dict=False)
    print(report_CE)
    cm = confusion_matrix(test_out, test_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1'])
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig('./Figures/Test_NN_CE.png')
    plt.show()
    