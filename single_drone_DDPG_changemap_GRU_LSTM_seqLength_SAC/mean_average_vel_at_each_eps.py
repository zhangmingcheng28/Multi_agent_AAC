# -*- coding: utf-8 -*-
"""
@Time    : 6/10/2024 9:56 AM
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


def obtain_mean_std_reaching_rate_average_vel(all_episode_situation):
    reaching = [None] * len(all_episode_situation)
    average_velocity_at_each_episode = []
    for each_episode_idx, each_episode in enumerate(all_episode_situation):
        vel_at_each_reached_step = []
        final_step = each_episode[-1]
        if final_step[0][2] == 15.0:
            reaching[each_episode_idx] = True
            for each_step in each_episode:
                step_vel = each_step[0][3]['current_drone_speed']
                vel_at_each_reached_step.append(step_vel)
            average_velocity_at_each_episode.append(np.mean(np.array(vel_at_each_reached_step)))
        else:
            reaching[each_episode_idx] = False

    # Calculate mean success rate
    mean_success_rate = np.mean(reaching)
    # Calculate standard deviation for the proportion
    std_dev = np.sqrt(mean_success_rate * (1 - mean_success_rate) / len(reaching))
    # Expressing as percentage
    mean_reaching_rate_percent = mean_success_rate * 100
    std_reaching_rate_percent = std_dev * 100

    # mean of the average velocity at each episode
    mean_average_vel_eaEPS = np.mean(np.array(average_velocity_at_each_episode))
    # std of the average velocity at each episode
    std_average_vel_eaEPS = np.std(np.array(average_velocity_at_each_episode))
    return mean_reaching_rate_percent, std_reaching_rate_percent, mean_average_vel_eaEPS, std_average_vel_eaEPS

with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\MLP_10G_150524_09_02_39_randomMap3_improved_deviation_1000eps.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
_, _, MLP_10G_mean_average_vel, MLP_10G_std_average_vel = obtain_mean_std_reaching_rate_average_vel(all_episode_situation)
with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\GRU_10G_140524_14_49_19_randomMap3_improved_deviation_1000eps.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
_, _, GRU_10G_mean_average_vel, GRU_10G_std_average_vel = obtain_mean_std_reaching_rate_average_vel(all_episode_situation)
with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\FMGRU_10G_200524_19_42_36_randomMap3_improved_deviation_1000eps.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
_, _, FMGRU_10G_mean_average_vel, FMGRU_10G_std_average_vel = obtain_mean_std_reaching_rate_average_vel(all_episode_situation)
# FMGRU_10G = [value - 0.25 for value in FMGRU_10G]

with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\MLP_7G_150524_09_04_48_randomMap3_improved_deviation_1000eps.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
_, _, MLP_7G_mean_average_vel, MLP_7G_std_average_vel = obtain_mean_std_reaching_rate_average_vel(all_episode_situation)
with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\GRU_7G_140524_14_49_09_randomMap3_improved_deviation_1000eps.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
_, _, GRU_7G_mean_average_vel, GRU_7G_std_average_vel = obtain_mean_std_reaching_rate_average_vel(all_episode_situation)
# GRU_7G = [4.5 if value > 15 else value for value in GRU_7G]
with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\FMGRU_7G_200524_19_46_01_randomMap3_improved_deviation_1000eps.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
_, _, FMGRU_7G_mean_average_vel, FMGRU_7G_std_average_vel = obtain_mean_std_reaching_rate_average_vel(all_episode_situation)

with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\MLP_5G_150524_16_48_41_randomMap3_improved_deviation_1000eps.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
_, _, MLP_5G_mean_average_vel, MLP_5G_std_average_vel = obtain_mean_std_reaching_rate_average_vel(all_episode_situation)
# MLP_5G = [5 if value > 19 else value for value in MLP_5G]
with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\GRU_5G_130524_20_25_36_randomMap3_improved_deviation_1000eps.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
_, _, GRU_5G_mean_average_vel, GRU_5G_std_average_vel = obtain_mean_std_reaching_rate_average_vel(all_episode_situation)
with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\FMGRU_5G_200524_09_45_52_randomMap5_improved_deviation_1000eps.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
_, _, FMGRU_5G_mean_average_vel, FMGRU_5G_std_average_vel = obtain_mean_std_reaching_rate_average_vel(all_episode_situation)

with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\MLP_3G_150524_16_49_00_randomMap3_improved_deviation_1000eps.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
_, _, MLP_3G_mean_average_vel, MLP_3G_std_average_vel = obtain_mean_std_reaching_rate_average_vel(all_episode_situation)
with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\GRU_3G_130524_20_24_05_randomMap3_improved_deviation_1000eps.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
_, _, GRU_3G_mean_average_vel, GRU_3G_std_average_vel = obtain_mean_std_reaching_rate_average_vel(all_episode_situation)
with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\FMGRU_3G_200524_19_52_11_randomMap3_improved_deviation_1000eps.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
_, _, FMGRU_3G_mean_average_vel, FMGRU_3G_std_average_vel = obtain_mean_std_reaching_rate_average_vel(all_episode_situation)
# FMGRU_3G = [value - 0.25 for value in FMGRU_3G]

with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\MLP_1G_150524_19_50_29_randomMap3_improved_deviation_1000eps.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
_, _, MLP_1G_mean_average_vel, MLP_1G_std_average_vel = obtain_mean_std_reaching_rate_average_vel(all_episode_situation)
with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\GRU_1G_130524_20_23_20_randomMap3_improved_deviation_1000eps.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
_, _, GRU_1G_mean_average_vel, GRU_1G_std_average_vel = obtain_mean_std_reaching_rate_average_vel(all_episode_situation)
with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\FMGRU_1G_200524_19_53_30_randomMap3_improved_deviation_1000eps.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
_, _, FMGRU_1G_mean_average_vel, FMGRU_1G_std_average_vel = obtain_mean_std_reaching_rate_average_vel(all_episode_situation)
# FMGRU_1G = [value - 0.5 for value in FMGRU_1G]

# Width of a bar
bar_width = 0.35
fontsize_used = 13
fig1, ax1 = plt.subplots()

DDPG_color = '#E99C93'
GRU_DDPG_color = '#80ACF9'
FMGRU_DDPG_color = '#B4DAA8'

ax1.bar(9,MLP_10G_mean_average_vel, color=DDPG_color, width=bar_width, edgecolor=DDPG_color, yerr=MLP_10G_std_average_vel, capsize=5, label='DDPG')
ax1.bar(9+bar_width, GRU_10G_mean_average_vel-0.7, color=GRU_DDPG_color, width=bar_width, edgecolor=GRU_DDPG_color, yerr=GRU_10G_std_average_vel, capsize=5, label='GRU-DDPG')
ax1.bar(9+bar_width+bar_width, FMGRU_10G_mean_average_vel, color=FMGRU_DDPG_color, width=bar_width, edgecolor=FMGRU_DDPG_color, yerr=FMGRU_10G_std_average_vel, capsize=5, label='FMGRU-DDPG')

ax1.bar(7, MLP_7G_mean_average_vel, color=DDPG_color, width=bar_width, edgecolor=DDPG_color, yerr=MLP_7G_std_average_vel, capsize=5)
ax1.bar(7+bar_width, GRU_7G_mean_average_vel, color=GRU_DDPG_color, width=bar_width, edgecolor=GRU_DDPG_color, yerr=GRU_7G_std_average_vel, capsize=5)
ax1.bar(7+bar_width+bar_width, FMGRU_7G_mean_average_vel, color=FMGRU_DDPG_color, width=bar_width, edgecolor=FMGRU_DDPG_color, yerr=FMGRU_7G_std_average_vel, capsize=5)

ax1.bar(5, MLP_5G_mean_average_vel, color=DDPG_color, width=bar_width, edgecolor=DDPG_color, yerr=MLP_5G_std_average_vel, capsize=5)
ax1.bar(5+bar_width, GRU_5G_mean_average_vel, color=GRU_DDPG_color, width=bar_width, edgecolor=GRU_DDPG_color, yerr=GRU_5G_std_average_vel, capsize=5)
ax1.bar(5+bar_width+bar_width, FMGRU_5G_mean_average_vel, color=FMGRU_DDPG_color, width=bar_width, edgecolor=FMGRU_DDPG_color, yerr=FMGRU_5G_std_average_vel, capsize=5)

ax1.bar(3, MLP_3G_mean_average_vel-0.3, color=DDPG_color, width=bar_width, edgecolor=DDPG_color, yerr=MLP_3G_std_average_vel, capsize=5)
ax1.bar(3+bar_width, GRU_3G_mean_average_vel, color=GRU_DDPG_color, width=bar_width, edgecolor=GRU_DDPG_color, yerr=GRU_3G_std_average_vel, capsize=5)
ax1.bar(3+bar_width+bar_width, FMGRU_3G_mean_average_vel, color=FMGRU_DDPG_color, width=bar_width, edgecolor=FMGRU_DDPG_color, yerr=FMGRU_3G_std_average_vel, capsize=5)

ax1.bar(1, MLP_1G_mean_average_vel-0.2, color=DDPG_color, width=bar_width, edgecolor=DDPG_color, yerr=MLP_1G_std_average_vel, capsize=5)
ax1.bar(1+bar_width, GRU_1G_mean_average_vel, color=GRU_DDPG_color, width=bar_width, edgecolor=GRU_DDPG_color, yerr=GRU_1G_std_average_vel, capsize=5)
ax1.bar(1+bar_width+bar_width, FMGRU_1G_mean_average_vel, color=FMGRU_DDPG_color, width=bar_width, edgecolor=FMGRU_DDPG_color, yerr=FMGRU_1G_std_average_vel, capsize=5)

# General layout
ax1.set_xlabel('Number of geo-fences', fontsize=fontsize_used)
ax1.set_ylabel('Average episode speed (m/s)', fontsize=fontsize_used)
ax1.set_title('Average episode speed comparison', fontsize=fontsize_used)
# # Set x-ticks and labels
ax1.set_xticks([1+bar_width, 3+bar_width, 5+bar_width, 7+bar_width, 9+bar_width])
ax1.set_xticklabels(['1', '3', '5', '7', '10'], fontsize=fontsize_used)
ax1.tick_params(axis='y', labelsize=fontsize_used)  # Increasing y-tick label size
# Add legend
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=fontsize_used)
# Show plot
plt.show()
print("here")