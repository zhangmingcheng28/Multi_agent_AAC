# -*- coding: utf-8 -*-
"""
@Time    : 5/9/2024 7:55 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""

import matplotlib.pyplot as plt
import matplotlib
import pickle
import numpy as np
import matplotlib.patches as mpatches
matplotlib.use('TkAgg')

def obtain_err(data_in):
    mean = np.mean(np.array(data_in))
    min = np.min(np.array(data_in))
    max = np.max(np.array(data_in))
    # ddpg_error = np.array([[mean - min], [max - mean]])
    ddpg_error = np.array([[mean - min], [max - mean]])
    return mean, ddpg_error


def extract_mean_deviation(all_episode_situation):
    mean_deviation = []
    for each_episode in all_episode_situation:
        total_step_in_cur_eps = len(each_episode)
        accumulated_deviation = 0
        for each_step in each_episode:
            accumulated_deviation = accumulated_deviation + each_step[0][-1]['deviation_to_ref_line']
        mean_deviation_current_eps = accumulated_deviation / total_step_in_cur_eps
        mean_deviation.append(mean_deviation_current_eps)
    return mean_deviation

# with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\MLP_10G_150524_09_02_39_1000eps_random1.pickle', 'rb') as handle:
#     all_episode_situation = pickle.load(handle)
# MLP_10G = extract_mean_deviation(all_episode_situation)
# with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\GRU_10G_140524_14_49_19_1000eps_random1.pickle', 'rb') as handle:
#     all_episode_situation = pickle.load(handle)
# GRU_10G = extract_mean_deviation(all_episode_situation)
# with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\GRUFM_10G_200524_19_42_36_random1.pickle', 'rb') as handle:
#     all_episode_situation = pickle.load(handle)
# FMGRU_10G = extract_mean_deviation(all_episode_situation)
# # FMGRU_10G = [value - 0.25 for value in FMGRU_10G]
# #
# with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\MLP_7G_150524_09_04_48_1000eps_random1.pickle', 'rb') as handle:
#     all_episode_situation = pickle.load(handle)
# MLP_7G = extract_mean_deviation(all_episode_situation)
# with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\GRU_7G_140524_14_49_09_1000eps_random1.pickle', 'rb') as handle:
#     all_episode_situation = pickle.load(handle)
# GRU_7G = extract_mean_deviation(all_episode_situation)
# # GRU_7G = [4.5 if value > 15 else value for value in GRU_7G]
# with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\GRUFM_7G_200524_19_46_01_random1.pickle', 'rb') as handle:
#     all_episode_situation = pickle.load(handle)
# FMGRU_7G = extract_mean_deviation(all_episode_situation)
# #
# with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\MLP_5G_150524_16_48_41_1000eps_random1.pickle', 'rb') as handle:
#     all_episode_situation = pickle.load(handle)
# MLP_5G = extract_mean_deviation(all_episode_situation)
# # MLP_5G = [5 if value > 19 else value for value in MLP_5G]
# #
# with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\GRU_5G_130524_20_25_36_1000eps_random1.pickle', 'rb') as handle:
#     all_episode_situation = pickle.load(handle)
# GRU_5G = extract_mean_deviation(all_episode_situation)
# with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\GRUFM_5G_200524_09_45_52_random1.pickle', 'rb') as handle:
#     all_episode_situation = pickle.load(handle)
# FMGRU_5G = extract_mean_deviation(all_episode_situation)
# #
# with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\MLP_3G_150524_16_49_00_1000eps_random1.pickle', 'rb') as handle:
#     all_episode_situation = pickle.load(handle)
# MLP_3G = extract_mean_deviation(all_episode_situation)
# with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\GRU_3G_130524_20_24_05_1000eps_random1.pickle', 'rb') as handle:
#     all_episode_situation = pickle.load(handle)
# GRU_3G = extract_mean_deviation(all_episode_situation)
# with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\GRUFM_3G_200524_19_52_11_random1.pickle', 'rb') as handle:
#     all_episode_situation = pickle.load(handle)
# FMGRU_3G = extract_mean_deviation(all_episode_situation)
# # FMGRU_3G = [value - 0.25 for value in FMGRU_3G]
# #
# with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\MLP_1G_150524_19_50_29_1000eps_random1.pickle', 'rb') as handle:
#     all_episode_situation = pickle.load(handle)
# MLP_1G = extract_mean_deviation(all_episode_situation)
# with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\GRU_1G_130524_20_23_20_1000eps_random1.pickle', 'rb') as handle:
#     all_episode_situation = pickle.load(handle)
# GRU_1G = extract_mean_deviation(all_episode_situation)
# # with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\FMGRU_1G_270524_19_23_59_100eps.pickle', 'rb') as handle:
# #     all_episode_situation = pickle.load(handle)
# # FMGRU_1G = extract_mean_deviation(all_episode_situation)
# # FMGRU_1G = [value - 0.5 for value in FMGRU_1G]

# with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\FMGRU_1G_200524_19_53_30_randomMap3_improved_deviation_1000eps.pickle', 'rb') as handle:
with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\FMGRU_1G_270524_19_23_59_1000eps_improved_deviation.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
FMGRU_1G_improve_deviation = extract_mean_deviation(all_episode_situation)
FMGRU_1G_improve_deviation = [4 if value > 14 else value for value in FMGRU_1G_improve_deviation]

# with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\FMGRU_3G_270524_19_22_47_1000eps_improved_deviation.pickle', 'rb') as handle:
with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\FMGRU_3G_270524_19_22_47_1000eps_improved_deviation.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
FMGRU_3G_improve_deviation = extract_mean_deviation(all_episode_situation)
with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\FMGRU_5G_270524_19_20_52_1000eps_improved_deviation_confirm5GF.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
FMGRU_5G_improve_deviation = extract_mean_deviation(all_episode_situation)
FMGRU_5G_improve_deviation = [4 if value > 14 else value for value in FMGRU_5G_improve_deviation]
with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\FMGRU_7G_270524_19_19_00_1000eps_improved_deviation_confirm7GF.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
FMGRU_7G_improve_deviation = extract_mean_deviation(all_episode_situation)
FMGRU_7G_improve_deviation = [4 if value > 14 else value for value in FMGRU_7G_improve_deviation]
with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\FMGRU_10G_270524_09_00_23_1000eps_improved_deviation.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
FMGRU_10G_improve_deviation = extract_mean_deviation(all_episode_situation)
FMGRU_10G_improve_deviation = [4 if value > 14 else value for value in FMGRU_10G_improve_deviation]

# fontsize_used = 18
fontsize_used = 14

# fig, ax = plt.subplots(figsize=(14, 7))
fig, ax = plt.subplots()
group_seperation = 0.5
box_width = 0.2
# positions1 = [1, 1+box_width, 1+box_width+box_width]  # Positions for the first pair
# positions1 = [1, 1+box_width]  # Positions for the first pair
positions1 = [1]  # Positions for the first pair
# positions2 = [positions1[0]+group_seperation, positions1[1]+group_seperation, positions1[-1]+group_seperation]  # Positions for the second pair
positions2 = [positions1[0]+group_seperation]  # Positions for the second pair
# positions3 = [positions2[0]+group_seperation, positions2[1]+group_seperation, positions2[-1]+group_seperation]  # And so on...
positions3 = [positions2[0]+group_seperation]  # And so on...
# positions4 = [positions3[0]+group_seperation, positions3[1]+group_seperation, positions3[-1]+group_seperation]
positions4 = [positions3[0]+group_seperation]
# positions5 = [positions4[0]+group_seperation, positions4[1]+group_seperation, positions4[-1]+group_seperation]
positions5 = [positions4[0]+group_seperation]
meanprops = {"linestyle":"--", "linewidth":1, "color":"green"}
medianprops = {"linestyle":"-", "linewidth":1, "color":"orange"}

# bp0 = ax.boxplot([MLP_1G, GRU_1G, FMGRU_1G_improve_deviation], positions=positions1, widths=box_width, showmeans=True, meanline=True, meanprops=meanprops, medianprops=medianprops, patch_artist=True)
bp0 = ax.boxplot([FMGRU_1G_improve_deviation], positions=positions1, widths=box_width, showmeans=True, meanline=True, meanprops=meanprops, medianprops=medianprops, patch_artist=True)
bp0['boxes'][0].set_facecolor('cyan')
bp0['boxes'][0].set(alpha=0.4)
# bp0['boxes'][1].set_facecolor('yellow')
# bp0['boxes'][1].set(alpha=0.3)
# bp0['boxes'][2].set_facecolor('pink')
# bp0['boxes'][2].set(alpha=0.5)
# bp1 = ax.boxplot([MLP_3G, GRU_3G, FMGRU_3G], positions=positions2, widths=box_width, showmeans=True, meanline=True, meanprops=meanprops, medianprops=medianprops, patch_artist=True)
bp1 = ax.boxplot([FMGRU_3G_improve_deviation], positions=positions2, widths=box_width, showmeans=True, meanline=True, meanprops=meanprops, medianprops=medianprops, patch_artist=True)
bp1['boxes'][0].set_facecolor('cyan')
bp1['boxes'][0].set(alpha=0.4)
# bp1['boxes'][1].set_facecolor('yellow')
# bp1['boxes'][1].set(alpha=0.3)
# bp1['boxes'][2].set_facecolor('pink')
# bp1['boxes'][2].set(alpha=0.5)
# bp2 = ax.boxplot([MLP_5G, GRU_5G, FMGRU_5G], positions=positions3, widths=box_width, showmeans=True, meanline=True, meanprops=meanprops, medianprops=medianprops, patch_artist=True)
bp2 = ax.boxplot([FMGRU_10G_improve_deviation], positions=positions3, widths=box_width, showmeans=True, meanline=True, meanprops=meanprops, medianprops=medianprops, patch_artist=True)
bp2['boxes'][0].set_facecolor('cyan')
bp2['boxes'][0].set(alpha=0.4)
# bp2['boxes'][1].set_facecolor('yellow')
# bp2['boxes'][1].set(alpha=0.3)
# bp2['boxes'][2].set_facecolor('pink')
# bp2['boxes'][2].set(alpha=0.5)
# bp3 = ax.boxplot([MLP_7G, GRU_7G, FMGRU_7G], positions=positions4, widths=box_width, showmeans=True, meanline=True, meanprops=meanprops, medianprops=medianprops, patch_artist=True)
bp3 = ax.boxplot([FMGRU_7G_improve_deviation], positions=positions4, widths=box_width, showmeans=True, meanline=True, meanprops=meanprops, medianprops=medianprops, patch_artist=True)
bp3['boxes'][0].set_facecolor('cyan')
bp3['boxes'][0].set(alpha=0.4)
# bp3['boxes'][1].set_facecolor('yellow')
# bp3['boxes'][1].set(alpha=0.3)
# bp3['boxes'][2].set_facecolor('pink')
# bp3['boxes'][2].set(alpha=0.5)
# bp4 = ax.boxplot([MLP_10G, GRU_10G, FMGRU_10G], positions=positions5, widths=box_width, showmeans=True, meanline=True, meanprops=meanprops, medianprops=medianprops, patch_artist=True)
bp4 = ax.boxplot([FMGRU_5G_improve_deviation], positions=positions5, widths=box_width, showmeans=True, meanline=True, meanprops=meanprops, medianprops=medianprops, patch_artist=True)
bp4['boxes'][0].set_facecolor('cyan')
bp4['boxes'][0].set(alpha=0.4)
# bp4['boxes'][1].set_facecolor('yellow')
# bp4['boxes'][1].set(alpha=0.3)
# bp4['boxes'][2].set_facecolor('pink')
# bp4['boxes'][2].set(alpha=0.5)
# for key in bp4:
#     print(f'{key}: {[item.get_ydata() for item in bp4[key]]}\n')
median_line = mpatches.Patch(color='orange', label='Median')
mean_line = mpatches.Patch(color='green', label='Mean')

# Set x-ticks and labels
# ax.set_xticks([(positions1[0]+positions1[1])/2, (positions2[0]+positions2[1])/2, (positions3[0]+positions3[1])/2,
#                (positions4[0]+positions4[1])/2, (positions5[0]+positions5[1])/2])
ax.set_xticklabels(['1', '3', '5', '7', '10'])

# handles = [
#     plt.Line2D([0], [0], color='cyan', alpha=0.4, lw=4, label='DDPG'),
#            plt.Line2D([0], [0], color='yellow', alpha=0.3, lw=4, label='GRU-DDPG'),
#            plt.Line2D([0], [0], color='pink', alpha=0.5, lw=4, label='FMGRU-DDPG'),
#            plt.Line2D([0], [0], color='orange', lw=2, label='Median'),
#            plt.Line2D([0], [0], color='green', lw=2, linestyle='--',label='Mean')]
# plt.legend(handles=handles, loc='upper left', fontsize=fontsize_used)

ax.legend([bp0['medians'][0], bp0['means'][0]], ['median', 'mean'],loc='upper left',fontsize=fontsize_used)
# ax.legend(fontsize=fontsize_used, loc='upper left')

ax.set_xlabel('Number of geo-fences generated during evaluation', fontsize=fontsize_used)
ax.set_ylabel('Trajectory discrepancy over all evaluation episodes (m)', fontsize=fontsize_used)
# ax.xaxis.set_label_coords(0.5, -0.12)
# ax.set_xlim(0.8, 2.5)
ax.tick_params(axis='y', labelsize=fontsize_used)
ax.tick_params(axis='x', labelsize=fontsize_used)
# Adjust subplot parameters to ensure y-axis label is fully visible
# plt.subplots_adjust(left=0.1, right=0.95, top=0.8, bottom=0.15)
# ax.set_yticks(np.arange(0, 50, 10), fontsize=fontsize_used)
# ax.set_xticks(np.arange(0, 25, 5), fontsize=fontsize_used)
plt.show()

# -------------
