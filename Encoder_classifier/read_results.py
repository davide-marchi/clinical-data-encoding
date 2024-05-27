import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.interpolate import griddata
from numpy import unique
from itertools import product



# Read JSON file
with open('Encoder_classifier/Models/results.json', 'r') as file:
    results = json.load(file)

X_unique = unique(sorted([model['embedding_perc'] for model in results]))
Y_unique = unique(sorted([model['masked_percentage'] for model in results]))

X_list = []
Y_list = []
Z_list = []

for embedding_perc, masked_percentage in product(X_unique, Y_unique):
    # check if this is working correctly
    max_f1_score = max([model['report']['macro avg']['f1-score'] for model in results if model['embedding_perc'] == embedding_perc and model['masked_percentage'] == masked_percentage])
    Z_list.append(max_f1_score)
    X_list.append(embedding_perc)
    Y_list.append(masked_percentage)
    
#Z_list = [model['report']['macro avg']['f1-score'] for model in results]

X, Y = np.meshgrid(X_unique, Y_unique)

# Interpola i valori Z per la griglia
Z = griddata((X_list, Y_list), Z_list, (X, Y), method='nearest') # TODO: Controllare se funzioni

#X, Y, Z = axes3d.get_test_data(0.05)

print(X)
print(Y)
print(Z)

ax = plt.figure().add_subplot(projection='3d')

# Plot the 3D surface
ax.plot_surface(X, Y, Z, edgecolor='royalblue', alpha=0.3)

# Plot projections of the contours for each dimension.  By choosing offsets
# that match the appropriate axes limits, the projected contours will sit on
# the 'walls' of the graph
ax.contourf(X, Y, Z, zdir='z', offset=-0.2, cmap='coolwarm')
ax.contourf(X, Y, Z, zdir='x', offset=-0.2, cmap='coolwarm')
ax.contourf(X, Y, Z, zdir='y', offset=-0.2, cmap='coolwarm')

ax.set(xlim=(-0.2, 1.2), ylim=(-0.2, 1.2), zlim=(0.6, 0.8), xlabel='Embedding percentage', ylabel='Masked percentage', zlabel='Macro avg F1-score')

plt.savefig('Encoder_classifier/Models/3dPlot.png')
plt.show()