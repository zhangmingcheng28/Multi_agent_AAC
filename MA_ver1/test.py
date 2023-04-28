# -*- coding: utf-8 -*-
"""
@Time    : 3/2/2023 10:26 AM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

# Generate some random data
x = np.random.normal(size=1000)
y = np.random.normal(size=1000)

# Set the number of bins for x and y
num_bins = 20

# Create the 2D histogram
plt.hist2d(x, y, bins=num_bins)

# Set the x-axis and y-axis labels
plt.xlabel('X')
plt.ylabel('Y')

# Set the title of the histogram
plt.title('2D Histogram of X and Y')

# Add a colorbar
plt.colorbar()

# Show the histogram
plt.show()