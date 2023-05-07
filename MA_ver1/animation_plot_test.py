# -*- coding: utf-8 -*-
"""
@Time    : 5/6/2023 4:12 PM
@Author  : mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import sys
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt
import time
import numpy as np
import random
import copy
import pickle
import matplotlib

matplotlib.use('TkAgg')
plt.ion()

done = 0
step = 1

# Define x and y arrays
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create figure and axis objects
fig, ax = plt.subplots()

# Plot the curve line
ax.plot(x, y, '--')


red_star_x, red_star_y = x[-1], y[-1]
green_triangle_x, green_triangle_y = x[0], y[0]
ax.plot(red_star_x, red_star_y, marker='*', color='red', markersize=10, linestyle='None')
ax.plot(green_triangle_x, green_triangle_y, marker='^', color='green', markersize=10, linestyle='None')

# Create scatter dot at the starting position
point, = ax.plot(x[0]+random.uniform(-3, 3), y[0]+random.uniform(-3, 3), 'o')

# Loop through the x and y arrays to animate the scatter dot
for i in range(len(x)):
    # Update the position of the scatter dot
    point.set_xdata(x[i])
    point.set_ydata(y[i]+random.uniform(-0.1, 0.1))

    # Redraw the figure and flush the events
    fig.canvas.draw()
    fig.canvas.flush_events()

    # Pause for 0.01 seconds
    time.sleep(0.01)

    # Clear the axis for the next frame
    ax.cla()

    # Plot the curve line again
    ax.plot(x, y, '--')
    ax.plot(red_star_x, red_star_y, marker='*', color='red', markersize=10, linestyle='None')
    ax.plot(green_triangle_x, green_triangle_y, marker='^', color='green', markersize=10, linestyle='None')
    # Plot the scatter dot at the new position
    ax.plot(point.get_xdata(), point.get_ydata(), 'o')

# Show the final figure
plt.show()