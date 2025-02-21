import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

import json
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from utilsData import unpack_encoder_name
from xgboost import XGBClassifier
from src.xgb_models.utility_xgb import encode_with_model, get_dataset # for encoding the data

years_to_death = 8

CLASSIFER_ONLY = True
TEST_SET = True

results = []
bestScore = 0
bestModel = None
n_estimators = 0
learning_rate = 0

path_to_results = './src/xgb_models/results'
if CLASSIFER_ONLY:
    path_to_results += '_C'
    if TEST_SET:
        path_to_results += '_T'
    else:
        path_to_results += '_V'
    path_to_results += f'_{years_to_death}Y.json'
else:
    path_to_results = f'./src/xgb_models/results{years_to_death}_Y.txt'

print(f'Loading results from {path_to_results}')
with open(path_to_results, 'r') as f:
    results = json.load(f)

sorted_results = sorted(results, key=lambda x: x['results']['macro avg']['recall'], reverse=True)
position = 1
bestModel = sorted_results[position]['encoder']
bestScore = sorted_results[position]['results']['macro avg']['recall']
best_result = sorted_results[position]['results']
n_estimators = sorted_results[position]['n_estimators']
learning_rate = sorted_results[position]['learning_rate']

if not CLASSIFER_ONLY and TEST_SET:
    dict_ = get_dataset()
    tr_data, ts_data, tr_out, ts_out = dict_['tr_data'], dict_['test_data'], dict_['tr_out'], dict_['test_out']
    tr_data_enc, ts_data_enc = encode_with_model(bestModel, tr_data, ts_data)
    xgb_model = XGBClassifier(
        n_estimators=n_estimators, 
        learning_rate=learning_rate, 
        n_jobs=2, 
        random_state=42,
        objective='binary:logistic'
    )
    xgb_model.fit(tr_data_enc, tr_out)
    ts_pred = xgb_model.predict(ts_data_enc)
    best_result = classification_report(ts_out, ts_pred, output_dict=True)
    bestScore = best_result['macro avg']['recall']


print(f'Best model: {bestModel}')
print(f'Best score: {bestScore}')
if not CLASSIFER_ONLY:
    print("_"*50)
    print(json.dumps(unpack_encoder_name(bestModel), indent=2))
print("_"*50)
print(f'Classifier hyperparameters: n_estimators={n_estimators}, learning_rate={learning_rate}')
print("_"*50)

#sorted_results = sorted(results, key=lambda x: x['results']['macro avg']['f1-score'], reverse=True)
#print("Sorted results by score:")
#for result in sorted_results[0:10]:
#    print(f"Model: {unpack_encoder_name(result['encoder'])['emb_perc']}, Score: {result['results']['macro avg']['f1-score']}")


# Estrai i valori della matrice di confusione
true_labels = []
pred_labels = []

for label in ["0.0", "1.0"]:  # Itera sulle classi
    support = int(best_result[label]["support"])  # Campioni reali per questa classe
    recall = best_result[label]["recall"]  # Recall = TP / (TP + FN)

    tp = int(support * recall)  # True Positives stimati
    fn = support - tp  # False Negatives stimati

    precision = best_result[label]["precision"]  # Precision = TP / (TP + FP)
    fp = int(tp / precision - tp) if precision > 0 else 0  # False Positives stimati

    true_labels.extend([int(float(label))] * support)  # Etichette reali
    pred_labels.extend([int(float(label))] * tp + [1 - int(float(label))] * fn)  # Predizioni stimate

# Calcola e mostra la matrice di confusione
cm = confusion_matrix(true_labels, pred_labels, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"])
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix for {'' if CLASSIFER_ONLY else 'Encoder + '}Classifier")
plt.savefig(f"xgb_f1_{'C' if CLASSIFER_ONLY else'EC'}_{'T' if TEST_SET else 'V'}.png", dpi=300)
plt.show()