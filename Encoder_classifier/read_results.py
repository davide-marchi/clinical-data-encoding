import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from numpy import unique
from itertools import product
# from mpl_toolkits.mplot3d import axes3d

# Read JSON file
with open('Encoder_classifier/Models/results.json', 'r') as file:
    results = json.load(file)

# Select the unique values of the embedding_perc and masked_percentage
X_unique = unique(sorted([model['embedding_perc'] for model in results]))
Y_unique = unique(sorted([model['masked_percentage'] for model in results]))

# Initialize empty lists (embedding_perc, masked_percentage, macro avg f1_score)
X_list = []
Y_list = []
Z_list = []

# Cycle for every possible permutation of embedding_perc and masked_percentage
for embedding_perc, masked_percentage in product(X_unique, Y_unique):

    """
    # To test
    model_list = [model for model in results if model['embedding_perc'] == embedding_perc and model['masked_percentage'] == masked_percentage]
    print(len(model_list))
    model_list_json = json.dumps(model_list, indent=4)
    with open('Encoder_classifier/Models/temp_list.json', 'w') as file:
        file.write(model_list_json)
    """

    # TODO: Check this
    max_f1_score = max([model['report']['macro avg']['f1-score'] for model in results if model['embedding_perc'] == embedding_perc and model['masked_percentage'] == masked_percentage])
    Z_list.append(max_f1_score)
    X_list.append(embedding_perc)
    Y_list.append(masked_percentage)

X, Y = np.meshgrid(X_unique, Y_unique)

# Interpola i valori Z per la griglia
Z = griddata((X_list, Y_list), Z_list, (X, Y), method='nearest') # TODO: Controllare se funzioni

#X, Y, Z = axes3d.get_test_data(0.05)

#print(X)
#print(Y)
#print(Z)

X_min = np.min(X)
X_max = np.max(X)
Y_min = np.min(Y)
Y_max = np.max(Y)
Z_min = np.min(Z)
Z_max = np.max(Z)
print(f"X min: {X_min}, X max: {X_max}")
print(f"Y min: {Y_min}, Y max: {Y_max}")
print(f"Z min: {Z_min}, Z max: {Z_max}")

ax = plt.figure().add_subplot(projection='3d')

# Plot the 3D surface
ax.plot_surface(X, Y, Z, edgecolor='royalblue', alpha=0.3)

# Plot projections of the contours for each dimension.  By choosing offsets
# that match the appropriate axes limits, the projected contours will sit on
# the 'walls' of the graph
ax.contourf(X, Y, Z, zdir='x', offset=np.min(X)-0.2, cmap='coolwarm')
ax.contourf(X, Y, Z, zdir='y', offset=np.min(Y)-0.2, cmap='coolwarm')
ax.contourf(X, Y, Z, zdir='z', offset=np.min(Z)-0.01, cmap='coolwarm')

ax.set(xlim=(np.min(X)-0.2, np.max(X)+0.2), ylim=(np.min(Y)-0.2, np.max(Y)+0.2), zlim=(np.min(Z)-0.01, np.max(Z)+0.01), xlabel='Embedding percentage', ylabel='Masked percentage', zlabel='Macro avg F1-score')

plt.savefig('Encoder_classifier/3dPlot.png')
plt.show()