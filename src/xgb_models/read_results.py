import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import json
from utilsData import unpack_encoder_name

years_to_death = 8

results = []
bestScore = 0
with open(f'./src/xgb_models/results{years_to_death}_Y.txt', 'r') as f:
    results = json.load(f)
for result in results:
    if result['results']['macro avg']['f1-score'] > bestScore:
        bestModel = result['encoder']
        bestScore = result['results']['macro avg']['f1-score']

print(f'Best model: {bestModel}')
print(f'Best score: {bestScore}')
print(json.dumps(unpack_encoder_name(bestModel), indent=2))

sorted_results = sorted(results, key=lambda x: x['results']['macro avg']['f1-score'], reverse=True)

print("Sorted results by score:")
for result in sorted_results[0:10]:
    print(f"Model: {unpack_encoder_name(result['encoder'])['emb_perc']}, Score: {result['results']['macro avg']['f1-score']}")