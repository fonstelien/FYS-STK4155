import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sys

file_name = sys.argv[1]

# Load the terrain
terrain = imread(file_name)
# Show the terrain
plt.figure()
plt.title(file_name)
plt.imshow(terrain, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
