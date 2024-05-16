import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.interpolate import griddata



# Read JSON file
with open('Encoder_classifier/Models/results.json', 'r') as file:
    results = json.load(file)

ax = plt.figure().add_subplot(projection='3d')
X_list = [t[0] for t in results]
Y_list = [t[1] for t in results]
Z_list = [t[3] for t in results] 

X, Y = np.meshgrid(X_list, Y_list)

# Interpola i valori Z per la griglia
Z = griddata((X_list, Y_list), Z_list, (X, Y), method='nearest') # TODO: Controllare se funzioni



#X, Y, Z = axes3d.get_test_data(0.05)

print(X)
print(Y)
print(Z)


# Plot the 3D surface
ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,
                alpha=0.3)

# Plot projections of the contours for each dimension.  By choosing offsets
# that match the appropriate axes limits, the projected contours will sit on
# the 'walls' of the graph
ax.contourf(X, Y, Z, zdir='z', offset=-100, cmap='coolwarm')
ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap='coolwarm')
ax.contourf(X, Y, Z, zdir='y', offset=40, cmap='coolwarm')

ax.set(xlim=(-40, 40), ylim=(-40, 40), zlim=(-100, 100),
       xlabel='X', ylabel='Y', zlabel='Z')

plt.show()