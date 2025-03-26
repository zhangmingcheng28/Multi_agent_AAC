# -*- coding: utf-8 -*-
"""
@Time    : 5/27/2024 5:26 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')


# Define the trajectory of the drone
trajectory = np.array([[0, 0], [1, 2], [2, 4], [3, 6], [4, 5], [5, 4], [6, 2], [7, 0], [8, -2], [9, -3]])
time_steps = range(len(trajectory))

# Define obstacles (appear at t=5)
obstacles = {'t=5': [5, 4]}

# Plotting
fig, ax = plt.subplots(figsize=(14, 7))

# Plot the trajectory with circles and lines linking them
for t, (x, y) in enumerate(trajectory):
    color = plt.cm.gray(t / len(trajectory))  # Gradually darkening color
    ax.plot(x, y, 'o', color=color, markersize=10)
    ax.text(x, y + 0.3, f't={t}', fontsize=9, ha='center', va='center', color='black')

# Draw lines between the points
ax.plot(trajectory[:, 0], trajectory[:, 1], color='black', linestyle='-', linewidth=1)

# Plot the obstacles
for label, (x, y) in obstacles.items():
    ax.plot(x, y, 's', color='red', markersize=15)
    ax.text(x + 0.5, y, label, fontsize=9, ha='center', va='center', color='blue')  # Shifted obstacle label

# Adding titles and labels
ax.set_title('Drone Trajectory with Time Steps and Obstacles')
ax.set_xlabel('X-coordinate')
ax.set_ylabel('Y-coordinate')

# Display the plot
plt.grid(True)
plt.axis('equal')
plt.show()