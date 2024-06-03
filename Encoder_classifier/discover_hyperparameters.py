import json
import numpy as np
import matplotlib.pyplot as plt
from numpy import unique
from itertools import product
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utilsData import unpack_encoder_name, unpack_classifier_name

# CLASSIFIER PARAMETERS
CL_batch_size = [200]
CL_learning_rate = [0.0001, 0.0002, 0.0004]
CL_plot = False
CL_weight_decay = [0.2e-5, 0.5e-5]
CL_num_epochs = [50]
CL_patience = [5]
CL_loss_weight = [(0.3, 0.7), (0.25, 0.75), (0.5, 0.5)]
# ENCODER PARAMETERS
EN_binary_loss_weight = [None, 0.5]
EN_batch_size = [200]
EN_learning_rate = [0.0015, 0.002]
EN_plot = False
EN_embedding_perc_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
EN_weight_decay = [0.05e-5, 0.2e-5]
EN_num_epochs = [250]
EN_masked_percentage_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
EN_patience = [10]

# Read JSON file
results_file = 'Encoder_classifier/sorted_results.json'
if not os.path.exists(results_file):
    # Sort results.json and store it
    with open('Encoder_classifier/results.json', 'r') as file:
        results = json.load(file)
    sorted_results = sorted(results, key=lambda x: x['report']['macro avg']['f1-score'], reverse=True)
    with open(results_file, 'w') as file:
        json.dump(sorted_results, file)
else:
    with open(results_file, 'r') as file:
        results = json.load(file)
        
surface_models = []
en_stats = {}
cl_stats = {}

for model in results:
    encoder_name = unpack_encoder_name(model['encoder'])
    classifier_name = unpack_classifier_name(model['classifier'])
    for key,value in encoder_name.items():
        en_stats[(key, value)] = 0
    for key,value in classifier_name.items():
        cl_stats[(key, value)] = 0
    
# Cycle for every possible permutation of embedding_perc and masked_percentage
for embedding_perc, masked_percentage in product(EN_embedding_perc_list, EN_masked_percentage_list):
    for model in results:
        if model['embedding_perc'] == embedding_perc and model['masked_percentage'] == masked_percentage:
            surface_models.append(model)
            break

# Sort surface_models by macro f1-score in descending order
surface_models.sort(key=lambda x: x['report']['macro avg']['f1-score'], reverse=True)

for model in surface_models:
    encoder_name = unpack_encoder_name(model['encoder'])
    classifier_name = unpack_classifier_name(model['classifier'])
    f1_score = model['report']['macro avg']['f1-score']
    print(f"Encoder Name: {json.dumps(encoder_name, indent=4)}, Classifier Name: {json.dumps(classifier_name, indent=4)}, F1 Score: {f1_score}")
    for key,value in encoder_name.items():
        en_stats[(key, value)] += 1
    for key,value in classifier_name.items():
        cl_stats[(key, value)] += 1

print("Encoder Stats")

# Order the couple key values in en_stats by values
en_stats_ordered = sorted(en_stats.items(), key=lambda x: x[1], reverse=True)

# Print the ordered en_stats
for key, value in en_stats_ordered:
    print(f"Key: {key}, Value: {value}")    

print("CLassifier Stats")

# Order the couple key values in en_stats by values
cl_stats_ordered = sorted(cl_stats.items(), key=lambda x: x[1], reverse=True)

# Print the ordered en_stats
for key, value in cl_stats_ordered:
    print(f"Key: {key}, Value: {value}")

# Calculate the sums of numbers with the same first value of the tuple used as keys
en_sums = {}
cl_sums = {}

for key, value in en_stats_ordered:
    first_value = key[0]
    if first_value not in en_sums:
        en_sums[first_value] = 0
    en_sums[first_value] += value

for key, value in cl_stats_ordered:
    first_value = key[0]
    if first_value not in cl_sums:
        cl_sums[first_value] = 0
    cl_sums[first_value] += value

"""
# Print the sums
print("Encoder Sums")
for key, value in en_sums.items():
    print(f"Key: {key}, Sum: {value}")

print("Classifier Sums")
for key, value in cl_sums.items():
    print(f"Key: {key}, Sum: {value}")
"""