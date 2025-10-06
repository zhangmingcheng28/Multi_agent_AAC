"""
@Time    : 13/8/2024 1:28 PM
@Author  : Bizhao Pang
@FileName:
@Description: plot all results for the special issue paper
@___Parameters___ to tune the training curve's shape and noise variance:
    1) window_size
    2) step
"""
import math
import matplotlib
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle

#-------------------------------IDDPG vs. single-agent DDPG-------------
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Set global font size
# plt.rcParams.update({'font.size': 14})
#
# # Data extracted from the tables
# aircraft = [2, 3, 4, 5, 6, 7, 8, 9, 10]
#
# # IDDPG data
# iddpg_goal = [100, 100, 100, 98, 98, 96, 95, 93, 93]
# iddpg_los = [0, 0, 0, 0, 0, 1, 0, 1, 2]
# iddpg_thunderstorm = [0, 0, 0, 2, 2, 3, 5, 6, 5]
# iddpg_distance_means = [1.07, 1.09, 1.08, 1.17, 1.17, 1.16, 1.17, 1.36, 1.38]
#
# # Single-agent DDPG data
# ddpg_goal = [100, 100, 97, 96, 90, 84, 83, 77, 47]
# ddpg_los = [0, 0, 0, 1, 2, 0, 1, 4, 4]
# ddpg_thunderstorm = [0, 0, 3, 3, 8, 16, 15, 19, 49]
# ddpg_distance_means = [1.11, 1.10, 1.10, 1.10, 1.11, 1.12, 1.12, 1.16, 1.12]
#
# # Prepare subplots
# fig, axes = plt.subplots(2, 2, figsize=(12, 9))
#
# # Subplot 1: Goal reach rate
# axes[0, 0].plot(aircraft, iddpg_goal, marker='o', label='IDDPG (proposed)')
# axes[0, 0].plot(aircraft, ddpg_goal, marker='s', label='Single-agent DDPG')
# axes[0, 0].set_title('(a) Goal reach rate')
# axes[0, 0].set_xlabel('Number of aircraft')
# axes[0, 0].set_ylabel('Rate (%)')
# axes[0, 0].set_xticks(range(2, 11))
# axes[0, 0].set_yticks(range(0, 101, 10))
# axes[0, 0].grid(True, linestyle='--', alpha=0.7)
# axes[0, 0].legend()
#
# # Subplot 2: Aircraft LOS rate
# axes[0, 1].plot(aircraft, iddpg_los, marker='o', label='IDDPG (proposed)')
# axes[0, 1].plot(aircraft, ddpg_los, marker='s', label='Single-agent DDPG')
# axes[0, 1].set_title('(b) Aircraft LOS rate')
# axes[0, 1].set_xlabel('Number of aircraft')
# axes[0, 1].set_ylabel('Rate (%)')
# axes[0, 1].set_xticks(range(2, 11))
# axes[0, 1].set_yticks(range(0, 6, 1))
# axes[0, 1].grid(True, linestyle='--', alpha=0.7)
# axes[0, 1].legend()
#
# # Subplot 3: Thunderstorm LOS rate
# axes[1, 0].plot(aircraft, iddpg_thunderstorm, marker='o', label='IDDPG (proposed)')
# axes[1, 0].plot(aircraft, ddpg_thunderstorm, marker='s', label='Single-agent DDPG')
# axes[1, 0].set_title('(c) Thunderstorm LOS rate')
# axes[1, 0].set_xlabel('Number of aircraft')
# axes[1, 0].set_ylabel('Rate (%)')
# axes[1, 0].set_xticks(range(2, 11))
# axes[1, 0].set_yticks(range(0, 51, 10))
# axes[1, 0].grid(True, linestyle='--', alpha=0.7)
# axes[1, 0].legend()
#
# # Subplot 4: Flight distance ratio (bar chart for means)
# x = np.arange(len(aircraft))  # Number of aircraft
# bar_width = 0.35
#
# axes[1, 1].bar(x - bar_width / 2, iddpg_distance_means, bar_width, label='IDDPG (proposed)', color='blue', alpha=0.7)
# axes[1, 1].bar(x + bar_width / 2, ddpg_distance_means, bar_width, label='Single-agent DDPG', color='orange', alpha=0.7)
#
# axes[1, 1].set_title('(d) Flight distance ratio')
# axes[1, 1].set_xlabel('Number of aircraft')
# axes[1, 1].set_ylabel('Distance ratio (mean)')
# axes[1, 1].set_xticks(x)
# axes[1, 1].set_xticklabels(aircraft)
# axes[1, 1].legend()
# axes[1, 1].grid(True, axis='y', linestyle='--', alpha=0.7)
#
# # Adjust layout
# plt.tight_layout()
# plt.show()
#
#
# print('end')



#------------------------------------Reward ablation study--------------
import matplotlib.pyplot as plt
import numpy as np

# Set global font size
# plt.rcParams.update({'font.size': 15})
#
# # Data from the table
# categories = ['Full model', 'No near aircraft penalty', 'No near storm penalty',
#               'No distance to goal', 'No heading change']
# metrics = ['Aircraft LOS rate', 'Thunderstorm LOS rate', 'Goal reach rate', 'Aircraft stray rate']
#
# data = np.array([
#     [0, 4, 4, 9, 1],  # Aircraft LOS rate
#     [2, 7, 11, 9, 2],  # Thunderstorm LOS rate
#     [98, 85, 78, 45, 97],  # Goal reach rate
#     [0, 4, 7, 37, 0]   # Aircraft stray rate
# ])
#
# # Colors for each bar
# colors = ['#1f77b4', '#ff7f0e', '#7f7f7f', '#d62728']  # Color-blind-friendly palette
#
# # Bar chart settings
# x = np.arange(len(categories))  # Number of groups
# bar_width = 0.15  # Width of each bar
#
# # Create the figure
# fig, ax = plt.subplots(figsize=(14, 7))
#
# # Plot each metric as a separate bar group
# for i in range(data.shape[0]):
#     ax.bar(
#         x + i * bar_width,
#         data[i],
#         width=bar_width,
#         color=colors[i],
#         edgecolor='black',
#         label=metrics[i],
#         alpha=0.8
#     )
#
# # Customize the plot
# ax.set_xlabel('Model variations', weight='bold')
# ax.set_ylabel('Percentage (%)')
# # ax.set_title('Comparison of metrics across model variations')
# ax.set_xticks(x + bar_width * (len(metrics) - 1) / 2)
# ax.set_xticklabels(categories, ha='center')  # Ensure labels are on a single line
# ax.set_ylim(0, 120)  # Set y-axis maximum to 100
#
# # Adjust legend position closer to the plot boundary
# fig.legend(
#     # title='Metrics',
#     loc='upper center',
#     bbox_to_anchor=(0.5, 0.85),  # Moved closer to the boundary
#     ncol=4,
#     frameon=False
# )
#
# # Add gridlines
# ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
#
# # Add data labels above bars
# for i in range(data.shape[0]):
#     for j in range(len(categories)):
#         ax.text(
#             x[j] + i * bar_width,
#             data[i][j] + 2,
#             f'{data[i][j]}%',
#             ha='center',
#             va='bottom'
#         )
#
# # Adjust layout for better spacing
# plt.tight_layout()
# plt.subplots_adjust(top=0.85)  # Adjust space for the legend at the top
#
#
# plt.show()
# print('end')

#---------------------ETA distributions--------------------
# # Define uncertainty levels with mean and standard deviation (converted to seconds)
# Set global font size
# plt.rcParams.update({'font.size': 12})
# uncertainty_levels = {
#     "Negligible": (0, 1.6 * 15),
#     "Low": (0, 6.4 * 15),
#     "Medium": (0, 14 * 15),
#     "High": (0, 30 * 15)
# }  # *15 converts to seconds, as one time step is 15 seconds
#
# # Sample size
# sample_size = 3000
#
# # Define subplot title labels
# subplot_labels = ['(a)', '(b)', '(c)', '(d)']
#
# # Create subplots for each uncertainty level
# fig, axes = plt.subplots(1, len(uncertainty_levels), figsize=(12, 4), sharey=True)
#
# # Custom x-ticks for each subplot
# custom_ticks = [
#     [-60, 0, 60],    # Negligible
#     [-300, 0, 300],  # Low
#     [-600, 0, 600],  # Medium
#     [-1200, 0, 1200] # High
# ]
#
# # Generate and plot histogram for each uncertainty level
# for ax, (subplot_label, (level, (mean, std_dev)), ticks) in zip(axes, zip(subplot_labels, uncertainty_levels.items(), custom_ticks)):
#     # Generate samples from a normal distribution
#     samples = np.random.normal(loc=mean, scale=std_dev, size=sample_size)
#
#     # Plot histogram
#     ax.hist(samples, bins=10, density=False, alpha=0.7, color='blue', edgecolor='black')
#
#     # Set titles and labels with subplot labels
#     ax.set_title(f"{subplot_label} {level} Uncertainty")
#     ax.set_xlabel("ETA uncertainty (s)")
#     ax.set_ylabel("Frequency")
#
#     # Set custom x-ticks
#     ax.set_xticks(ticks)
#     ax.set_xticklabels([str(tick) for tick in ticks])
#
# # Adjust layout for clarity
# plt.tight_layout()
# plt.show()
# print('end')




#________________Number of aircraft concurrently in the airspace (%)______________
# import numpy as np
# import pandas as pd
# import matplotlib
# import matplotlib.pyplot as plt
# from collections import Counter
# import pickle
#
# import matplotlib.pyplot as plt
# from collections import Counter
# import pickle
#
# # Set global font size
# plt.rcParams.update({'font.size': 14})
#
# # Load data for each uncertainty level
# with open(r'D:\MADDPG_2nd_jp\261124_16_37_03\interval_record_eps\v2_30AC_activated_AC.pickle', 'rb') as f:
#     total_ac_normal = pickle.load(f)
# with open(r'D:\MADDPG_2nd_jp\261124_16_37_03\interval_record_eps_low\v2_30AC_activated_AC.pickle', 'rb') as f:
#     total_ac_low = pickle.load(f)
# with open(r'D:\MADDPG_2nd_jp\261124_16_37_03\interval_record_eps_mid\v2_30AC_activated_AC.pickle', 'rb') as f:
#     total_ac_mid = pickle.load(f)
# with open(r'D:\MADDPG_2nd_jp\261124_16_37_03\interval_record_eps_high\v2_30AC_activated_AC.pickle', 'rb') as f:
#     total_ac_high = pickle.load(f)
#
#
# # Helper function to calculate percentages
# def calculate_percentages(data):
#     all_values = [value for sublist in data.values() for value in sublist]
#     counter = Counter(all_values)
#     total_count = sum(counter.values())
#     percentages = {k: (v / total_count) * 100 for k, v in counter.items()}
#     return dict(sorted(percentages.items()))
#
#
# # Calculate percentages for each uncertainty level
# percentages_normal = calculate_percentages(total_ac_normal)
# percentages_low = calculate_percentages(total_ac_low)
# percentages_mid = calculate_percentages(total_ac_mid)
# percentages_high = calculate_percentages(total_ac_high)
#
# # Plot settings
# fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=300)  # Higher resolution with 2x2 grid
# axes = axes.flatten()
#
# titles = [
#     '(a) Negligible Uncertainty',
#     '(b) Low Uncertainty',
#     '(c) Medium Uncertainty',
#     '(d) High Uncertainty'
# ]
# percentages = [percentages_normal, percentages_low, percentages_mid, percentages_high]
#
# # Plot each subplot
# for ax, title, data in zip(axes, titles, percentages):
#     bars = ax.barh(list(data.keys()), list(data.values()), color='gray', edgecolor='black', alpha=0.7, height=0.5)
#
#     # Adjust text position dynamically for each subplot
#     for bar in bars:
#         width = bar.get_width()
#         yloc = bar.get_y() + bar.get_height() / 2
#         x_limit = ax.get_xlim()[1]  # Get the upper x-axis limit
#
#         # Adjust text placement dynamically based on the bar width
#         if width < x_limit * 0.05:  # Very small bars
#             text_x = width + 0.5
#             ha = 'left'
#         elif width < x_limit * 0.9:  # Medium-sized bars
#             text_x = width + 0.5
#             ha = 'left'
#         else:  # Large bars
#             text_x = width - 2
#             ha = 'center'
#
#         ax.text(text_x, yloc, f'{width:.2f}%', va='center', ha=ha, color='black')
#     # Set title and labels
#     ax.set_title(title)
#     ax.xaxis.grid(True, linestyle='--', color='gray', alpha=0.5)
#     ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.5)
#     ax.set_xlabel('Number of aircraft concurrently in the airspace (%)')
#     ax.set_ylabel('No. of Aircraft')
#     # Ensure all y-axis labels are shown
#     ax.set_yticks(list(data.keys()))
#     ax.set_yticklabels(list(data.keys()))
#
# # Adjust layout with extra padding to prevent text cutoff
# plt.tight_layout()
# plt.show()

#---------------------actual separation under various uncertainty levels---------------
# Load data for each uncertainty level
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Set global font size
# plt.rcParams.update({'font.size': 15})
#
# import matplotlib.pyplot as plt
# import pickle
#
# # Load data from pickles
# with open(r'D:\MADDPG_2nd_jp\261124_16_37_03\interval_record_eps\v2_actual_eta.pickle', 'rb') as f:
#     total_ac_normal = pickle.load(f)
# with open(r'D:\MADDPG_2nd_jp\261124_16_37_03\interval_record_eps_low\v2_actual_eta.pickle', 'rb') as f:
#     total_ac_low = pickle.load(f)
# with open(r'D:\MADDPG_2nd_jp\261124_16_37_03\interval_record_eps_mid\v2_actual_eta.pickle', 'rb') as f:
#     total_ac_mid = pickle.load(f)
# with open(r'D:\MADDPG_2nd_jp\261124_16_37_03\interval_record_eps_high\v2_actual_eta.pickle', 'rb') as f:
#     total_ac_high = pickle.load(f)
#
# # Prepare data for plotting
# datasets = {
#     'Negligible': total_ac_normal,
#     'Low': total_ac_low,
#     'Medium': total_ac_mid,
#     'High': total_ac_high
# }
#
# # Plot each dataset
# plt.figure(figsize=(8, 6))
#
# for label, data in datasets.items():
#     # Flatten the lists for plotting (if data is a list of lists)
#     all_values = [value for sublist in data for value in sublist] if isinstance(data[0], list) else data
#
#     # convert time step to minutes, 12s/step
#     all_values = [value / 6 for value in all_values]
#
#     # Sort values from highest to lowest
#     sorted_values = sorted(all_values, reverse=True)
#
#     # Plot the sorted data
#     plt.plot(sorted_values, label=f"{label} uncertainty")
#
# # Add the reference line
# plt.axhline(y=10, color='black', linestyle='--', linewidth=2.5, label='Separation minima')
# plt.xlim(0, 3000)  # Set x-axis range to 0-3000
#
# # Add plot details
# # plt.title('Line Plot of Actual ETA Values (Sorted)')
# plt.xlabel('Numer of aircraft pairs')
# plt.ylabel('Actual separation between aircraft (min)')
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()
#
# print('end')


#-----------higher uncertainty level requires more time steps to clear a batch of ac-------
# Load data for each uncertainty level
# with open(r'D:\MADDPG_2nd_jp\261124_16_37_03\interval_record_eps\v2_30AC_activated_AC.pickle', 'rb') as f:
#     total_ac_normal = pickle.load(f)
# with open(r'D:\MADDPG_2nd_jp\261124_16_37_03\interval_record_eps_low\v2_30AC_activated_AC.pickle', 'rb') as f:
#     total_ac_low = pickle.load(f)
# with open(r'D:\MADDPG_2nd_jp\261124_16_37_03\interval_record_eps_mid\v2_30AC_activated_AC.pickle', 'rb') as f:
#     total_ac_mid = pickle.load(f)
# with open(r'D:\MADDPG_2nd_jp\261124_16_37_03\interval_record_eps_high\v2_30AC_activated_AC.pickle', 'rb') as f:
#     total_ac_high = pickle.load(f)
#
# import matplotlib.pyplot as plt
# matplotlib.rcParams.update({'font.size': 13})
# # Calculate the lengths of lists in each dictionary and remove lengths < 200
# def process_data(data_dict):
#     lengths = [len(lst) for lst in data_dict.values()]
#     filtered_lengths = [length / 5 for length in lengths if length >= 350]
#     return filtered_lengths
#
# # Process the data
# lengths_normal = process_data(total_ac_normal)
# lengths_low = process_data(total_ac_low)
# lengths_mid = process_data(total_ac_mid)
# lengths_high = process_data(total_ac_high)
#
# # Prepare data for box plot
# data_to_plot = [lengths_normal, lengths_low, lengths_mid, lengths_high]
# labels = ['Negligible', 'Low', 'Medium', 'High']
#
# # Plot the box plot
# plt.figure(figsize=(6, 5))
# plt.boxplot(data_to_plot, labels=labels, patch_artist=True, boxprops=dict(facecolor='skyblue', color='black'))
# # plt.title('Box Plot of List Lengths for Each Group')
# plt.xlabel('Uncertainty level')
# plt.ylabel('Total time for an batch of aircraft (min)')
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()
#
# print('end')

# ------------------------------ETA uncertainty---------------------------------------
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import pickle
# import matplotlib
#
# matplotlib.rcParams.update({'font.size': 14})
#
# with open(r'D:\MADDPG_2nd_jp\261124_16_37_03\interval_record_eps\_30AC_activated_AC.pickle', 'rb') as f:
#     total_ac = pickle.load(f)
#
# fig, ax = plt.subplots(figsize=(8, 6))
#
# for i, data in total_ac.items():
#     # ax.plot(data, label=f"List {i}")  # line graph
#     ax.scatter(range(len(data)), data, label=f"List {i}")  # Scatter plot
#
# ax.set_xlabel("Experiment time (minutes)")
# ax.set_ylabel("No. of aircraft in airspace")
#
# # Find the largest length among all lists
# max_length = max(len(lst) for lst in total_ac.values())
# rounded_max_length = math.ceil(max_length / 40) * 40 # Round up to the nearest multiple of 40
# # Customize the x-axis
# ticks_interval = 40  # Interval of 40 ticks
# tick_positions = list(range(0, rounded_max_length + 1, ticks_interval))
# tick_labels = [str(i * 10 // 40) for i in tick_positions]  # Convert ticks to minutes
# plt.xticks(ticks=tick_positions, labels=tick_labels)
#
# # Add dashed vertical lines as background
# for x in tick_positions:
#     ax.axvline(x=x, color='black', linestyle='--', linewidth=0.4, alpha=0.6)
#
# # Improve the layout
# plt.tight_layout()
# plt.show()
#
# print('end')
# ----------------------END--------ETA uncertainty---------------------------------------


# # Specify the file path
# file_path = r'D:\MADDPG_2nd_jp\training curves.xlsx'
#
# # Read the data from the CSV file, no header since the file contains only reward values
# data = pd.read_excel(file_path, header=None)
#
# # Convert all data to numeric, forcing errors to NaN (you can skip this step if not needed)
# data = data.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
#
# # Convert the DataFrame to a numpy array for easier manipulation
# rewards = data.values  # No need to transpose if shape is (4, 16384)
#
# # Smoothing function (moving average)
# def moving_average(data, window_size=100):
#     return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
#
# # Apply smoothing
# smoothed_rewards = np.array([moving_average(reward, window_size=500) for reward in rewards])
#
# # Calculate mean and confidence intervals for smoothed data
# mean_rewards = smoothed_rewards.mean(axis=1)
# std_rewards = smoothed_rewards.std(axis=1)
#
# # X-axis values (episodes)
# episodes = np.arange(1, len(smoothed_rewards[0]) + 1, step=200)  # Plot every 10th episode
# smoothed_rewards = smoothed_rewards[:, ::200]
# # Plotting
# plt.figure(figsize=(8, 6))
#
# # Model 1
# plt.plot(episodes, smoothed_rewards[0], 'r-', label='Model 1')
# plt.fill_between(episodes, smoothed_rewards[0] - std_rewards[0], smoothed_rewards[0] + std_rewards[0], color='red', alpha=0.2)
#
# # Model 2
# plt.plot(episodes, smoothed_rewards[1], 'navy', label='Model 2')
# plt.fill_between(episodes, smoothed_rewards[1] - std_rewards[1], smoothed_rewards[1] + std_rewards[1], color='navy', alpha=0.2)
#
# # Model 3
# plt.plot(episodes, smoothed_rewards[2], 'purple', label='Model 3')
# plt.fill_between(episodes, smoothed_rewards[2] - std_rewards[2], smoothed_rewards[2] + std_rewards[2], color='purple', alpha=0.2)
#
# # Model 4
# plt.plot(episodes, smoothed_rewards[3], 'brown', label='Model 4')
# plt.fill_between(episodes, smoothed_rewards[3] - std_rewards[3], smoothed_rewards[3] + std_rewards[3], color='brown', alpha=0.2)
#
# # Adding labels and title
# plt.xlabel('Training episode')
# plt.ylabel('Reward')
# plt.title('Training Curve with Confidence Interval')
# plt.legend()
#
# # Show plot
# plt.show()
#
#

#___________plot training curves______________v1.0
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Specify the file path
# file_path = r'D:\MADDPG_2nd_jp\training curves.xlsx'
#
# # Read the data from the Excel file, no header since the file contains only reward values
# data = pd.read_excel(file_path, header=None)
#
# # Convert all data to numeric, forcing errors to NaN (you can skip this step if not needed)
# data = data.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
#
# # Transpose the DataFrame since your data is now (20000, 4) instead of (4, 20000)
# # “-5000” is to move y-axis, “/50” is to reduce magnitude effect
# rewards = (data.values.T - 4000)/50  # Transpose the data to get shape (4, 20000)
#
# # Smoothing function (moving average)
# def moving_average(data, window_size=200):
#     return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
#
# # Apply smoothing
# smoothed_rewards = np.array([moving_average(reward, window_size=2000) for reward in rewards])
#
# # Calculate mean and confidence intervals for smoothed data
# mean_rewards = smoothed_rewards.mean(axis=1)
# std_rewards = smoothed_rewards.std(axis=1)
#
# # X-axis values (episodes)
# episodes = np.arange(1, len(smoothed_rewards[0]) + 1, step=50)  # Plot every 200th episode
# smoothed_rewards = smoothed_rewards[:, ::50]
#
# # Set global font size
# plt.rcParams.update({'font.size': 12})
#
# # Plotting
# plt.figure(figsize=(8, 8))
#
# # Model 1
# plt.plot(episodes, smoothed_rewards[0], 'r-', label='IDDPG')
# plt.fill_between(episodes, smoothed_rewards[0] - std_rewards[0], smoothed_rewards[0] + std_rewards[0], color='red', alpha=0.2)
#
# # Model 2
# plt.plot(episodes, smoothed_rewards[1], 'navy', label='IDDPG-n')
# plt.fill_between(episodes, smoothed_rewards[1] - std_rewards[1], smoothed_rewards[1] + std_rewards[1], color='navy', alpha=0.2)
#
# # Model 3
# plt.plot(episodes, smoothed_rewards[2], 'purple', label='IDDPG-s')
# plt.fill_between(episodes, smoothed_rewards[2] - std_rewards[2], smoothed_rewards[2] + std_rewards[2], color='purple', alpha=0.2)
#
# # Model 4
# plt.plot(episodes, smoothed_rewards[3], 'g', label='IDDPG-ns')
# plt.fill_between(episodes, smoothed_rewards[3] - std_rewards[3], smoothed_rewards[3] + std_rewards[3], color='brown', alpha=0.2)
#
# # Adding labels and title
# plt.xlabel('Training episode')
# plt.ylabel('Reward')
# # plt.title('Training Curve with Confidence Interval')
# plt.legend()
#
# # Show plot
# plt.show()

#___________plot training curves______________v1.0

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Specify the file path
file_path = (r'C:\Users\18322\OneDrive - Nanyang Technological University\[1] '
             r'Research Papers\[1] Journals\AST 2025\Data and results\training curves.xlsx')

# Load data for each uncertainty level
with open(r'D:\MADDPG_2nd_jp\260625_13_50_19\toplot\all_episode_reward.pickle', 'rb') as f:
    total_ac_normal1 = pickle.load(f)

with open(r'D:\MADDPG_2nd_jp\260625_16_57_16\toplot\all_episode_reward.pickle', 'rb') as f:
    total_ac_normal2 = pickle.load(f)

# Read the data from the Excel file, no header since the file contains only reward values
data = pd.read_excel(file_path, header=None)

# Convert all data to numeric, forcing errors to NaN (you can skip this step if not needed)
data = data.applymap(lambda x: pd.to_numeric(x, errors='coerce'))

#___________________________________ processing reward _______________________________
# Transpose the DataFrame since your data is now (20000, 4) instead of (4, 20000)
# "-500" is to move y-axis, "/20" is to reduce magnitude effect
# rewards = (data.values.T - 500)/20  # Transpose the data to get shape (4, 20000)
# 1. Count number of steps per episode (only first 20000)
step_counts_1 = [len(episode) for episode in total_ac_normal1[:20000]]
step_counts_2 = [len(episode) for episode in total_ac_normal2[:20000]]

# 2. Extract rewards for first 20000 episodes
episode_rewards_1 = data.iloc[:20000, 0].values  # First simulation
episode_rewards_2 = data.iloc[:20000, 1].values  # Second simulation

# 3. Compute average reward per step
ave_reward_1 = episode_rewards_1 / np.array(step_counts_1)
ave_reward_2 = episode_rewards_2 / np.array(step_counts_2)
# 4. Stack the result to match shape (2, 30000)
rewards = np.vstack([ave_reward_1, ave_reward_2])
# rewards = (data.values.T - 500)/20  # Transpose the data to get shape (4, 20000)

# Smoothing function (moving average)
def moving_average(data, window_size=1000):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Apply smoothing
smoothed_rewards = np.array([moving_average(reward, window_size=1000) for reward in rewards])

# Calculate mean and confidence intervals for smoothed data
mean_rewards = smoothed_rewards.mean(axis=1)
std_rewards = smoothed_rewards.std(axis=1)

# X-axis values (episodes)
step_size = 50
episodes = np.arange(1, len(smoothed_rewards[0]) + 1, step=step_size)  # Plot every 200th episode
smoothed_rewards = smoothed_rewards[:, ::step_size]

# Set global font size
plt.rcParams.update({'font.size': 16})

# Plotting
plt.figure(figsize=(11, 6))  # Slightly increased figure size for better readability

# Model 1
plt.plot(episodes, smoothed_rewards[0], 'r-', label='DDPG-v200')
plt.fill_between(episodes, smoothed_rewards[0] - std_rewards[0], smoothed_rewards[0] + std_rewards[0], color='red', alpha=0.2)

# Model 2
plt.plot(episodes, smoothed_rewards[1], 'navy', label='DDPG-v300')
plt.fill_between(episodes, smoothed_rewards[1] - std_rewards[1], smoothed_rewards[1] + std_rewards[1], color='navy', alpha=0.2)


# Customize x-axis labels to display as 2k, 4k, 6k, ..., 20k
# plt.xticks(ticks=np.arange(0, 20001, 2000), labels=[f'{x//1000+1}k' for x in np.arange(0, 20001-800, 2000-80)])
plt.xticks(ticks=np.arange(0, 20001, 2000))
# xticks = np.arange(0, 20001, 2000)
# xtick_labels = [f'{x//1000}k' if x >= 1000 else f'{x}' for x in xticks]

# Adding labels and title
plt.xlabel('Training episode', fontsize=16)
plt.ylabel('Average reward per time step', fontsize=16)
plt.legend()

# Improve the layout
plt.tight_layout()

# Show plot
plt.show()




#_______________plot box figure using mean and std____________v1.0
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Simulate data based on provided means and standard deviations
# np.random.seed(42)  # For reproducibility
#
# # Number of aircraft
# aircraft_counts = [4, 5, 6, 7, 8]
#
# # Means and standard deviations for each group
# means = [1.0754, 1.1656, 1.1744, 1.1613, 1.1725]
# std_devs = [0.0566, 0.1479, 0.1639, 0.1691, 0.1743]
#
# # Simulate data for each aircraft count
# data = [np.random.normal(mean, std_dev, 100) for mean, std_dev in zip(means, std_devs)]
#
# # Set global font size
# plt.rcParams.update({'font.size': 14})
# # Adding labels and title
#
# # Create box plot
# plt.figure(figsize=(8, 6))
# plt.boxplot(data, labels=aircraft_counts, patch_artist=True)
#
#
# plt.xlabel('Number of aircraft')
# plt.ylabel('Flight distance ratio')
# # plt.title('Scalability Analysis: Flight Distance Ratio with Increased Aircraft')
#
# # Show plot
# plt.show()


# v2.0___in use
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.lines as mlines
#
# # Simulate data based on provided means and standard deviations
# np.random.seed(42)  # For reproducibility
#
# # Number of aircraft
# aircraft_counts = [4, 5, 6, 7, 8]
#
# # Means and standard deviations for each group
# means = [1.0754, 1.1656, 1.1744, 1.1613, 1.1725]
# std_devs = [0.0566, 0.1479, 0.1639, 0.1691, 0.1743]
#
# # Simulate data for each aircraft count
# data = [np.random.normal(mean, std_dev, 100) for mean, std_dev in zip(means, std_devs)]
#
# # Set global font size
# plt.rcParams.update({'font.size': 14})
#
# # Create box plot with additional features
# plt.figure(figsize=(8, 6))
# box = plt.boxplot(data, labels=aircraft_counts, patch_artist=True, notch=True,
#                   showmeans=True, meanline=True, whiskerprops=dict(linewidth=2))
#
# # Customizing the box plot appearance
# colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink', 'lightgrey']
# for patch, color in zip(box['boxes'], colors):
#     patch.set_facecolor(color)
#
# # Adding mean markers
# for mean in box['means']:
#     mean.set(marker='o', color='black', markersize=3)
#
# # Adding median lines
# for median in box['medians']:
#     median.set(color='red', linewidth=2)
#
# # Customizing whiskers and caps
# for whisker in box['whiskers']:
#     whisker.set(color='black', linestyle='-', linewidth=2)
#
# for cap in box['caps']:
#     cap.set(color='black', linewidth=2)
#
# # Adding labels and title
# plt.xlabel('Number of aircraft')
# plt.ylabel('Flight distance ratio')
# # plt.title('Scalability Analysis: Flight Distance Ratio with Increased Aircraft')
#
# # Adding a grid for better readability
# plt.grid(True, linestyle='--', alpha=0.7)
#
# # Create custom legend
# mean_line = mlines.Line2D([], [], color='g', marker='o', linestyle='None', markersize=3, label='Mean')
# median_line = mlines.Line2D([], [], color='red', linewidth=2, label='Median')
#
# plt.legend(handles=[mean_line, median_line], loc='upper left')
# # Show plot
# plt.show()


#__________plot rubostness under different weathers_______v1.0
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Set the font to "Times New Roman" globally
# plt.rcParams.update({
#     'font.family': 'serif',
#     'font.serif': ['Times New Roman'],
#     'text.usetex': False
# })
# # Data from Table 7
# scenarios = ['c_1S', 'c_2S', 'c_3S', 'c_1L', 'c_2L', 'c_3L']
# collision_rate_aircraft = [1, 0, 3, 4, 4, 3]
# collision_rate_storm_cell = [5, 12, 11, 10, 24, 34]
# goal_reach_rate = [94, 88, 86, 86, 72, 63]
# flight_distance_ratio = [1.1436, 1.2205, 1.2593, 1.1955, 1.2915, 1.2471]
#
# # Set up the figure and axes
# fig, axes = plt.subplots(2, 2, figsize=(8, 6))
# # fig.suptitle('Simulation Results Under Diverse Weather Scenarios', fontsize=16, fontweight='bold')
#
# # Set a subtle background color
# # fig.patch.set_facecolor('#f2f2f2')
# # for ax in axes.flat:
# #     ax.set_facecolor('#f7f7f7')
#
# # Bar width
# bar_width = 0.4
# index = np.arange(len(scenarios))
#
# # Define a color palette with transparency
# colors = [[0.121, 0.466, 0.705, 0.8], [1.0, 0.498, 0.054, 0.8], [0.173, 0.627, 0.173, 0.8], [0.839, 0.153, 0.157, 0.8]]
#
# # Plotting Collision Rate with Aircraft
# axes[0, 0].bar(index, collision_rate_aircraft, bar_width, color=colors[0], edgecolor='black')
# axes[0, 0].set_xlabel('Scenario', fontsize=12)
# axes[0, 0].set_ylabel('Aircraft LOS rate (%)', fontsize=12)
# axes[0, 0].set_xticks(index)
# axes[0, 0].set_xticklabels(scenarios, fontsize=12)
# axes[0, 0].tick_params(axis='y', labelsize=12)
# axes[0, 0].spines['top'].set_visible(False)
# axes[0, 0].spines['right'].set_visible(False)
# axes[0, 0].set_title('Aircraft LOS rate', fontsize=12, fontweight='bold')
#
# # Plotting Collision Rate with Storm Cells
# axes[0, 1].bar(index, collision_rate_storm_cell, bar_width, color=colors[1], edgecolor='black')
# axes[0, 1].set_xlabel('Scenario', fontsize=12)
# axes[0, 1].set_ylabel('Thunderstorm LOS rate (%)', fontsize=12)
# axes[0, 1].set_xticks(index)
# axes[0, 1].set_xticklabels(scenarios, fontsize=12)
# axes[0, 1].tick_params(axis='y', labelsize=12)
# axes[0, 1].spines['top'].set_visible(False)
# axes[0, 1].spines['right'].set_visible(False)
# axes[0, 1].set_title('Thunderstorm LOS rate', fontsize=12, fontweight='bold')
#
# # Plotting Goal Reach Rate
# axes[1, 0].bar(index, goal_reach_rate, bar_width, color=colors[2], edgecolor='black')
# axes[1, 0].set_xlabel('Scenario', fontsize=12)
# axes[1, 0].set_ylabel('Goal reach rate (%)', fontsize=12)
# axes[1, 0].set_xticks(index)
# axes[1, 0].set_xticklabels(scenarios, fontsize=12)
# axes[1, 0].tick_params(axis='y', labelsize=12)
# axes[1, 0].spines['top'].set_visible(False)
# axes[1, 0].spines['right'].set_visible(False)
# axes[1, 0].set_title('Goal reach rate', fontsize=12, fontweight='bold')
#
# # Plotting Flight Distance Ratio
# axes[1, 1].bar(index, flight_distance_ratio, bar_width, color=colors[3], edgecolor='black')
# axes[1, 1].set_xlabel('Scenario', fontsize=12)
# axes[1, 1].set_ylabel('Flight distance ratio', fontsize=12)
# axes[1, 1].set_xticks(index)
# axes[1, 1].set_xticklabels(scenarios, fontsize=12)
# axes[1, 1].tick_params(axis='y', labelsize=12)
# axes[1, 1].spines['top'].set_visible(False)
# axes[1, 1].spines['right'].set_visible(False)
# axes[1, 1].set_title('Flight distance ratio', fontsize=12, fontweight='bold')
#
# # Adjust layout for better spacing
# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.show()
