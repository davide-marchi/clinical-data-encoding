import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from utilsData import unpack_encoder_name

years_to_death = 8

CLASSIFER_ONLY = False

results = []
bestScore = 0
bestModel = None
n_estimators = 0
learning_rate = 0
if CLASSIFER_ONLY:
    with open(f'./src/xgb_models/results_C_{years_to_death}_Y.json', 'r') as f:
        results = json.load(f)
else:
    with open(f'./src/xgb_models/results{years_to_death}_Y.txt', 'r') as f:
        results = json.load(f)
for result in results:
    #balanced_accuracy = (result['results']["0.0"]["recall"] + result['results']["1.0"]["recall"]) / 2
    #result['results']['balanced_accuracy'] = balanced_accuracy
    #if balanced_accuracy > bestScore:
    if result['results']['macro avg']['f1-score'] > bestScore:
        bestModel = result['encoder']
        bestScore = result['results']['macro avg']['f1-score']
        #bestScore = balanced_accuracy
        best_result = result['results']
        n_estimators = result['n_estimators']
        learning_rate = result['learning_rate']

print(f'Best model: {bestModel}')
print(f'Best score: {bestScore}')
if not CLASSIFER_ONLY:
    print("_"*50)
    print(json.dumps(unpack_encoder_name(bestModel), indent=2))
print("_"*50)
print(f'Classifier hyperparameters: n_estimators={n_estimators}, learning_rate={learning_rate}')
print("_"*50)

sorted_results = sorted(results, key=lambda x: x['results']['macro avg']['f1-score'], reverse=True)

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
plt.title(f'Confusion Matrix for {'' if CLASSIFER_ONLY else 'Encoder + '}Classifier')
plt.savefig(f'xgb_f1_{'C' if CLASSIFER_ONLY else'EC'}.png', dpi=300)
plt.show()