import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utilsData import load_past_results_and_models, unpack_encoder_name
import json

results, _, _ = load_past_results_and_models()

bestModel:str = None
bestScore:float = 0

for result in results:
    if result['results']['macro avg']['f1-score'] > bestScore:
        bestModel = result['encoder_string']
        bestScore = result['results']['macro avg']['f1-score']

print(f'Best model: {bestModel}')
print(f'Best score: {bestScore}')
print(json.dumps(unpack_encoder_name(bestModel), indent=2))