import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utilsData import load_past_results_and_models, unpack_encoder_name
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import json

results, _, _ = load_past_results_and_models()

CLASSIFER_ONLY = False
bestModel:str = None
classifier_bestModel:dict = None
bestScore:float = 0

for result in results:
    if CLASSIFER_ONLY:
        if result['encoder_string'] == 'encoder_None_None_None_None_None_None_None_None':
            bestModel = result['encoder_string']
            bestScore = result['results']['macro avg']['f1-score']
            best_result = result['results']
            classifier_bestModel = result['classifier']
            break
        else:
            continue
    #balanced_accuracy = (result['results']["0.0"]["recall"] + result['results']["1.0"]["recall"]) / 2
    #result['results']['balanced_accuracy'] = balanced_accuracy
    #if balanced_accuracy > bestScore:
    if result['results']['macro avg']['f1-score'] > bestScore:
        bestModel = result['encoder_string']
        bestScore = result['results']['macro avg']['f1-score']
        #bestScore = balanced_accuracy
        best_result = result['results']
        classifier_bestModel = result['classifier']


print(f'Best model: {bestModel}')
print(f'Best score: {bestScore}')
if not CLASSIFER_ONLY:
    print("_"*50)
    print(json.dumps(unpack_encoder_name(bestModel), indent=2))
print("_"*50)
print('Classifier model hyperparameter: ')
print(json.dumps(classifier_bestModel, indent=2))
print("_"*50)

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
plt.title(f'Confusion Matrix for{'' if CLASSIFER_ONLY else ' Encoder +'} Classifier')
plt.savefig(f'nn_f1_{"C" if CLASSIFER_ONLY else "EC"}_loc.png', dpi=300)
plt.show()