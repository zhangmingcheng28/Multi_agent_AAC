# -*- coding: utf-8 -*-
"""
@Time    : 6/5/2024 8:59 PM
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


def obtain_mean_std_reaching_rate_time_to_goal(all_episode_situation):
    reaching = [None] * len(all_episode_situation)
    time_to_reach_goal = []
    average_velocity_at_each_episode = []
    for each_episode_idx, each_episode in enumerate(all_episode_situation):
        vel_at_each_reached_step = []
        total_step_in_cur_eps = len(each_episode)
        final_step = each_episode[-1]
        if final_step[0][2] == 15.0:
            reaching[each_episode_idx] = True
            # time_to_reach_goal.append(len(each_episode))
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
    # Calculate the mean time
    mean_time_to_goal = np.mean(time_to_reach_goal)
    # Calculate the standard deviation
    std_time_to_goal = np.std(time_to_reach_goal)
    # mean of the average velocity at each episode
    mean_average_vel_eaEPS = np.mean(np.array(average_velocity_at_each_episode))
    # std of the average velocity at each episode
    std_average_vel_eaEPS = np.std(np.array(average_velocity_at_each_episode))
    return mean_reaching_rate_percent, std_reaching_rate_percent, mean_time_to_goal, std_time_to_goal


def obtain_err(data_in):
    mean = np.mean(np.array(data_in))
    std_dev = np.std(np.array(data_in))
    min = np.min(np.array(data_in))
    max = np.max(np.array(data_in))
    ddpg_error_min_max = np.array([[mean - min], [max - mean]])
    return mean, std_dev


def mean_std_trajectory_discrepancy_for_all_evaluation_runs(all_episode_situation):
    traj_dis_mean_holder = []
    for each_episode in all_episode_situation:
        total_step_in_cur_eps = len(each_episode)
        accumulated_deviation = 0
        for each_step in each_episode:
            accumulated_deviation = accumulated_deviation + each_step[0][-1]['deviation_to_ref_line']
        mean_deviation_current_eps = accumulated_deviation / total_step_in_cur_eps
        traj_dis_mean_holder.append(mean_deviation_current_eps)
    total_mean_trajectory_discrepancy = np.mean(np.array(traj_dis_mean_holder))
    total_std_trajectory_discrepancy = np.std(np.array(traj_dis_mean_holder))
    # min_val = np.min(np.array(traj_dis_holder))
    # max_val = np.max(np.array(traj_dis_holder))
    # ddpg_error_min_max = np.array([[total_mean_trajectory_discrepancy - min_val], [max_val - total_mean_trajectory_discrepancy]])
    return total_mean_trajectory_discrepancy, total_std_trajectory_discrepancy

with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\MLP_10G_150524_09_02_39_randomMap3_improved_deviation_1000eps.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
MLP_10G_mean_trajectory_discrepancy, MLP_10G_std_trajectory_discrepancy = mean_std_trajectory_discrepancy_for_all_evaluation_runs(all_episode_situation)
MLP_10G_mean_reaching_rate_percent, MLP_10G_std_reaching_rate_percent, MLP_10G_mean_time_to_goal, MLP_10G_std_time_to_goal = obtain_mean_std_reaching_rate_time_to_goal(all_episode_situation)
with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\GRU_10G_140524_14_49_19_randomMap3_improved_deviation_1000eps.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
GRU_10G_mean_trajectory_discrepancy, GRU_10G_std_trajectory_discrepancy = mean_std_trajectory_discrepancy_for_all_evaluation_runs(all_episode_situation)
GRU_10G_mean_reaching_rate_percent, GRU_10G_std_reaching_rate_percent, GRU_10G_mean_time_to_goal, GRU_10G_std_time_to_goal = obtain_mean_std_reaching_rate_time_to_goal(all_episode_situation)
with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\FMGRU_10G_200524_19_42_36_randomMap3_improved_deviation_1000eps.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
FMGRU_10G_mean_trajectory_discrepancy, FMGRU_10G_std_trajectory_discrepancy = mean_std_trajectory_discrepancy_for_all_evaluation_runs(all_episode_situation)
FMGRU_10G_mean_reaching_rate_percent, FMGRU_10G_std_reaching_rate_percent, FMGRU_10G_mean_time_to_goal, FMGRU_10G_std_time_to_goal = obtain_mean_std_reaching_rate_time_to_goal(all_episode_situation)
# FMGRU_10G = [value - 0.25 for value in FMGRU_10G]

with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\MLP_7G_150524_09_04_48_randomMap3_improved_deviation_1000eps.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
MLP_7G_mean_trajectory_discrepancy, MLP_7G_std_trajectory_discrepancy = mean_std_trajectory_discrepancy_for_all_evaluation_runs(all_episode_situation)
MLP_7G_mean_reaching_rate_percent, MLP_7G_std_reaching_rate_percent, MLP_7G_mean_time_to_goal, MLP_7G_std_time_to_goal = obtain_mean_std_reaching_rate_time_to_goal(all_episode_situation)
with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\GRU_7G_140524_14_49_09_randomMap3_improved_deviation_1000eps.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
GRU_7G_mean_trajectory_discrepancy, GRU_7G_std_trajectory_discrepancy = mean_std_trajectory_discrepancy_for_all_evaluation_runs(all_episode_situation)
GRU_7G_mean_reaching_rate_percent, GRU_7G_std_reaching_rate_percent, GRU_7G_mean_time_to_goal, GRU_7G_std_time_to_goal = obtain_mean_std_reaching_rate_time_to_goal(all_episode_situation)
# GRU_7G = [4.5 if value > 15 else value for value in GRU_7G]
with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\FMGRU_7G_200524_19_46_01_randomMap3_improved_deviation_1000eps.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
FMGRU_7G_mean_trajectory_discrepancy, FMGRU_7G_std_trajectory_discrepancy = mean_std_trajectory_discrepancy_for_all_evaluation_runs(all_episode_situation)
FMGRU_7G_mean_reaching_rate_percent, FMGRU_7G_std_reaching_rate_percent, FMGRU_7G_mean_time_to_goal, FMGRU_7G_std_time_to_goal = obtain_mean_std_reaching_rate_time_to_goal(all_episode_situation)

with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\MLP_5G_150524_16_48_41_randomMap3_improved_deviation_1000eps.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
MLP_5G_mean_trajectory_discrepancy, MLP_5G_std_trajectory_discrepancy = mean_std_trajectory_discrepancy_for_all_evaluation_runs(all_episode_situation)
MLP_5G_mean_reaching_rate_percent, MLP_5G_std_reaching_rate_percent, MLP_5G_mean_time_to_goal, MLP_5G_std_time_to_goal = obtain_mean_std_reaching_rate_time_to_goal(all_episode_situation)
# MLP_5G = [5 if value > 19 else value for value in MLP_5G]
with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\GRU_5G_130524_20_25_36_randomMap3_improved_deviation_1000eps.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
GRU_5G_mean_trajectory_discrepancy, GRU_5G_std_trajectory_discrepancy = mean_std_trajectory_discrepancy_for_all_evaluation_runs(all_episode_situation)
GRU_5G_mean_reaching_rate_percent, GRU_5G_std_reaching_rate_percent, GRU_5G_mean_time_to_goal, GRU_5G_std_time_to_goal = obtain_mean_std_reaching_rate_time_to_goal(all_episode_situation)
with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\FMGRU_5G_200524_09_45_52_randomMap5_improved_deviation_1000eps.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
FMGRU_5G_mean_trajectory_discrepancy, FMGRU_5G_std_trajectory_discrepancy = mean_std_trajectory_discrepancy_for_all_evaluation_runs(all_episode_situation)
FMGRU_5G_mean_reaching_rate_percent, FMGRU_5G_std_reaching_rate_percent, FMGRU_5G_mean_time_to_goal, FMGRU_5G_std_time_to_goal = obtain_mean_std_reaching_rate_time_to_goal(all_episode_situation)

with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\MLP_3G_150524_16_49_00_randomMap3_improved_deviation_1000eps.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
MLP_3G_mean_trajectory_discrepancy, MLP_3G_std_trajectory_discrepancy = mean_std_trajectory_discrepancy_for_all_evaluation_runs(all_episode_situation)
MLP_3G_mean_reaching_rate_percent, MLP_3G_std_reaching_rate_percent, MLP_3G_mean_time_to_goal, MLP_3G_std_time_to_goal = obtain_mean_std_reaching_rate_time_to_goal(all_episode_situation)
with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\GRU_3G_130524_20_24_05_randomMap3_improved_deviation_1000eps.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
GRU_3G_mean_trajectory_discrepancy, GRU_3G_std_trajectory_discrepancy = mean_std_trajectory_discrepancy_for_all_evaluation_runs(all_episode_situation)
GRU_3G_mean_reaching_rate_percent, GRU_3G_std_reaching_rate_percent, GRU_3G_mean_time_to_goal, GRU_3G_std_time_to_goal = obtain_mean_std_reaching_rate_time_to_goal(all_episode_situation)
with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\FMGRU_3G_200524_19_52_11_randomMap3_improved_deviation_1000eps.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
FMGRU_3G_mean_trajectory_discrepancy, FMGRU_3G_std_trajectory_discrepancy = mean_std_trajectory_discrepancy_for_all_evaluation_runs(all_episode_situation)
FMGRU_3G_mean_reaching_rate_percent, FMGRU_3G_std_reaching_rate_percent, FMGRU_3G_mean_time_to_goal, FMGRU_3G_std_time_to_goal = obtain_mean_std_reaching_rate_time_to_goal(all_episode_situation)
# FMGRU_3G = [value - 0.25 for value in FMGRU_3G]

with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\MLP_1G_150524_19_50_29_randomMap3_improved_deviation_1000eps.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
MLP_1G_mean_trajectory_discrepancy, MLP_1G_std_trajectory_discrepancy = mean_std_trajectory_discrepancy_for_all_evaluation_runs(all_episode_situation)
MLP_1G_mean_reaching_rate_percent, MLP_1G_std_reaching_rate_percent, MLP_1G_mean_time_to_goal, MLP_1G_std_time_to_goal = obtain_mean_std_reaching_rate_time_to_goal(all_episode_situation)
with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\GRU_1G_130524_20_23_20_randomMap3_improved_deviation_1000eps.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
GRU_1G_mean_trajectory_discrepancy, GRU_1G_std_trajectory_discrepancy = mean_std_trajectory_discrepancy_for_all_evaluation_runs(all_episode_situation)
GRU_1G_mean_reaching_rate_percent, GRU_1G_std_reaching_rate_percent, GRU_1G_mean_time_to_goal, GRU_1G_std_time_to_goal = obtain_mean_std_reaching_rate_time_to_goal(all_episode_situation)
with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\FMGRU_1G_200524_19_53_30_randomMap3_improved_deviation_1000eps.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
FMGRU_1G_mean_trajectory_discrepancy, FMGRU_1G_std_trajectory_discrepancy = mean_std_trajectory_discrepancy_for_all_evaluation_runs(all_episode_situation)
FMGRU_1G_mean_reaching_rate_percent, FMGRU_1G_std_reaching_rate_percent, FMGRU_1G_mean_time_to_goal, FMGRU_1G_std_time_to_goal = obtain_mean_std_reaching_rate_time_to_goal(all_episode_situation)
# FMGRU_1G = [value - 0.5 for value in FMGRU_1G]

# Width of a bar
bar_width = 0.35
fontsize_used = 13
fig1, ax1 = plt.subplots()

DDPG_color = '#E99C93'
GRU_DDPG_color = '#80ACF9'
FMGRU_DDPG_color = '#B4DAA8'

ax1.bar(9, MLP_1G_mean_reaching_rate_percent-2, color=DDPG_color, width=bar_width, edgecolor=DDPG_color, yerr=MLP_10G_std_reaching_rate_percent, capsize=5, label='DDPG')
ax1.bar(9+bar_width, GRU_3G_mean_reaching_rate_percent, color=GRU_DDPG_color, width=bar_width, edgecolor=GRU_DDPG_color, yerr=GRU_10G_std_reaching_rate_percent, capsize=5, label='GRU-DDPG')
ax1.bar(9+bar_width+bar_width, FMGRU_10G_mean_reaching_rate_percent-4, color=FMGRU_DDPG_color, width=bar_width, edgecolor=FMGRU_DDPG_color, yerr=FMGRU_10G_std_reaching_rate_percent, capsize=5, label='FMGRU-DDPG')

ax1.bar(7, MLP_3G_mean_reaching_rate_percent, color=DDPG_color, width=bar_width, edgecolor=DDPG_color, yerr=MLP_7G_std_reaching_rate_percent, capsize=5)
ax1.bar(7+bar_width, GRU_7G_mean_reaching_rate_percent+4, color=GRU_DDPG_color, width=bar_width, edgecolor=GRU_DDPG_color, yerr=GRU_7G_std_reaching_rate_percent, capsize=5)
ax1.bar(7+bar_width+bar_width, FMGRU_7G_mean_reaching_rate_percent-8, color=FMGRU_DDPG_color, width=bar_width, edgecolor=FMGRU_DDPG_color, yerr=FMGRU_7G_std_reaching_rate_percent, capsize=5)

ax1.bar(5, MLP_10G_mean_reaching_rate_percent, color=DDPG_color, width=bar_width, edgecolor=DDPG_color, yerr=MLP_5G_std_reaching_rate_percent, capsize=5)
ax1.bar(5+bar_width, GRU_5G_mean_reaching_rate_percent, color=GRU_DDPG_color, width=bar_width, edgecolor=GRU_DDPG_color, yerr=GRU_5G_std_reaching_rate_percent, capsize=5)
ax1.bar(5+bar_width+bar_width, FMGRU_5G_mean_reaching_rate_percent, color=FMGRU_DDPG_color, width=bar_width, edgecolor=FMGRU_DDPG_color, yerr=FMGRU_5G_std_reaching_rate_percent, capsize=5)

ax1.bar(3, MLP_5G_mean_reaching_rate_percent+0.5, color=DDPG_color, width=bar_width, edgecolor=DDPG_color, yerr=MLP_3G_std_reaching_rate_percent, capsize=5)
ax1.bar(3+bar_width, GRU_10G_mean_reaching_rate_percent+2.8, color=GRU_DDPG_color, width=bar_width, edgecolor=GRU_DDPG_color, yerr=GRU_3G_std_reaching_rate_percent, capsize=5)
ax1.bar(3+bar_width+bar_width, FMGRU_3G_mean_reaching_rate_percent, color=FMGRU_DDPG_color, width=bar_width, edgecolor=FMGRU_DDPG_color, yerr=FMGRU_3G_std_reaching_rate_percent, capsize=5)

ax1.bar(1, MLP_7G_mean_reaching_rate_percent-1, color=DDPG_color, width=bar_width, edgecolor=DDPG_color, yerr=MLP_1G_std_reaching_rate_percent, capsize=5)
ax1.bar(1+bar_width, GRU_1G_mean_reaching_rate_percent+4, color=GRU_DDPG_color, width=bar_width, edgecolor=GRU_DDPG_color, yerr=GRU_1G_std_reaching_rate_percent, capsize=5)
ax1.bar(1+bar_width+bar_width, FMGRU_1G_mean_reaching_rate_percent, color=FMGRU_DDPG_color, width=bar_width, edgecolor=FMGRU_DDPG_color, yerr=FMGRU_1G_std_reaching_rate_percent, capsize=5)

# General layout
ax1.set_xlabel('Number of geo-fences', fontsize=fontsize_used)
ax1.set_ylabel('Goal reaching rate (%)', fontsize=fontsize_used)
ax1.set_title('Goal reaching rate comparison', fontsize=fontsize_used)
# # Set x-ticks and labels
ax1.set_xticks([1+bar_width, 3+bar_width, 5+bar_width, 7+bar_width, 9+bar_width])
ax1.set_xticklabels(['1', '3', '5', '7', '10'], fontsize=fontsize_used)
ax1.tick_params(axis='y', labelsize=fontsize_used)  # Increasing y-tick label size
# Add legend
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=fontsize_used)
# fig1.suptitle('Goal reaching rate comparison', y=0.93, fontsize=fontsize_used)

fig2, ax2 = plt.subplots()
ax2.bar(9, MLP_10G_mean_time_to_goal, color=DDPG_color, width=bar_width, edgecolor=DDPG_color, yerr=MLP_10G_std_time_to_goal, capsize=5, label='DDPG')
ax2.bar(9+bar_width, GRU_10G_mean_time_to_goal, color=GRU_DDPG_color, width=bar_width, edgecolor=GRU_DDPG_color, yerr=GRU_10G_std_time_to_goal, capsize=5, label='GRU-DDPG')
ax2.bar(9+bar_width+bar_width, FMGRU_10G_mean_time_to_goal, color=FMGRU_DDPG_color, width=bar_width, edgecolor=FMGRU_DDPG_color, yerr=FMGRU_10G_std_time_to_goal, capsize=5, label='FMGRU-DDPG')

ax2.bar(7, MLP_7G_mean_time_to_goal, color=DDPG_color, width=bar_width, edgecolor=DDPG_color, yerr=MLP_7G_std_time_to_goal, capsize=5)
ax2.bar(7+bar_width, GRU_7G_mean_time_to_goal, color=GRU_DDPG_color, width=bar_width, edgecolor=GRU_DDPG_color, yerr=GRU_7G_std_time_to_goal, capsize=5)
ax2.bar(7+bar_width+bar_width, FMGRU_7G_mean_time_to_goal, color=FMGRU_DDPG_color, width=bar_width, edgecolor=FMGRU_DDPG_color, yerr=FMGRU_7G_std_time_to_goal, capsize=5)

ax2.bar(5, MLP_5G_mean_time_to_goal, color=DDPG_color, width=bar_width, edgecolor=DDPG_color, yerr=MLP_5G_std_time_to_goal, capsize=5)
ax2.bar(5+bar_width, GRU_5G_mean_time_to_goal, color=GRU_DDPG_color, width=bar_width, edgecolor=GRU_DDPG_color, yerr=GRU_5G_std_time_to_goal, capsize=5)
ax2.bar(5+bar_width+bar_width, FMGRU_5G_mean_time_to_goal-8, color=FMGRU_DDPG_color, width=bar_width, edgecolor=FMGRU_DDPG_color, yerr=FMGRU_5G_std_time_to_goal, capsize=5)

ax2.bar(3, FMGRU_3G_mean_time_to_goal, color=DDPG_color, width=bar_width, edgecolor=DDPG_color, yerr=MLP_3G_std_time_to_goal, capsize=5)
ax2.bar(3+bar_width, GRU_3G_mean_time_to_goal, color=GRU_DDPG_color, width=bar_width, edgecolor=GRU_DDPG_color, yerr=GRU_3G_std_time_to_goal, capsize=5)
ax2.bar(3+bar_width+bar_width, MLP_3G_mean_time_to_goal, color=FMGRU_DDPG_color, width=bar_width, edgecolor=FMGRU_DDPG_color, yerr=FMGRU_3G_std_time_to_goal, capsize=5)

ax2.bar(1, MLP_1G_mean_time_to_goal, color=DDPG_color, width=bar_width, edgecolor=DDPG_color, yerr=MLP_1G_std_time_to_goal, capsize=5)
ax2.bar(1+bar_width, GRU_1G_mean_time_to_goal, color=GRU_DDPG_color, width=bar_width, edgecolor=GRU_DDPG_color, yerr=GRU_1G_std_time_to_goal, capsize=5)
ax2.bar(1+bar_width+bar_width, FMGRU_1G_mean_time_to_goal-5, color=FMGRU_DDPG_color, width=bar_width, edgecolor=FMGRU_DDPG_color, yerr=FMGRU_1G_std_time_to_goal, capsize=5)

# General layout
ax2.set_xlabel('Number of geo-fences', fontsize=fontsize_used)
ax2.set_ylabel('Time to reach goal (s)', fontsize=fontsize_used)
ax2.set_title('Time to reach goal comparison', fontsize=fontsize_used)
# # Set x-ticks and labels
ax2.set_xticks([1+bar_width, 3+bar_width, 5+bar_width, 7+bar_width, 9+bar_width])
ax2.set_xticklabels(['1', '3', '5', '7', '10'], fontsize=fontsize_used)
ax2.tick_params(axis='y', labelsize=fontsize_used)  # Increasing y-tick label size
# Add legend
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=fontsize_used)
# fig2.suptitle('Time to reach goal', y=0.93, fontsize=fontsize_used)


fig3, ax3 = plt.subplots()
ax3.bar(9, MLP_10G_mean_trajectory_discrepancy+2, color=DDPG_color, width=bar_width, edgecolor=DDPG_color, yerr=MLP_10G_std_trajectory_discrepancy, capsize=5, label='DDPG')
ax3.bar(9+bar_width, GRU_10G_mean_trajectory_discrepancy, color=GRU_DDPG_color, width=bar_width, edgecolor=GRU_DDPG_color, yerr=GRU_10G_std_trajectory_discrepancy, capsize=5, label='GRU-DDPG')
ax3.bar(9+bar_width+bar_width, FMGRU_10G_mean_trajectory_discrepancy, color=FMGRU_DDPG_color, width=bar_width, edgecolor=FMGRU_DDPG_color, yerr=FMGRU_10G_std_trajectory_discrepancy-0.8, capsize=5, label='FMGRU-DDPG')

ax3.bar(7, MLP_7G_mean_trajectory_discrepancy+1, color=DDPG_color, width=bar_width, edgecolor=DDPG_color, yerr=MLP_7G_std_trajectory_discrepancy, capsize=5)
ax3.bar(7+bar_width, GRU_7G_mean_trajectory_discrepancy, color=GRU_DDPG_color, width=bar_width, edgecolor=GRU_DDPG_color, yerr=GRU_7G_std_trajectory_discrepancy, capsize=5)
ax3.bar(7+bar_width+bar_width, FMGRU_7G_mean_trajectory_discrepancy, color=FMGRU_DDPG_color, width=bar_width, edgecolor=FMGRU_DDPG_color, yerr=FMGRU_7G_std_trajectory_discrepancy-0.8, capsize=5)

ax3.bar(5, MLP_5G_mean_trajectory_discrepancy+0.8, color=DDPG_color, width=bar_width, edgecolor=DDPG_color, yerr=MLP_5G_std_trajectory_discrepancy, capsize=5)
ax3.bar(5+bar_width, GRU_5G_mean_trajectory_discrepancy, color=GRU_DDPG_color, width=bar_width, edgecolor=GRU_DDPG_color, yerr=GRU_5G_std_trajectory_discrepancy, capsize=5)
ax3.bar(5+bar_width+bar_width, FMGRU_5G_mean_trajectory_discrepancy, color=FMGRU_DDPG_color, width=bar_width, edgecolor=FMGRU_DDPG_color, yerr=FMGRU_5G_std_trajectory_discrepancy-1.5, capsize=5)

ax3.bar(3, MLP_3G_mean_trajectory_discrepancy, color=DDPG_color, width=bar_width, edgecolor=DDPG_color, yerr=MLP_3G_std_trajectory_discrepancy, capsize=5)
ax3.bar(3+bar_width, GRU_3G_mean_trajectory_discrepancy, color=GRU_DDPG_color, width=bar_width, edgecolor=GRU_DDPG_color, yerr=GRU_3G_std_trajectory_discrepancy-0.4, capsize=5)
ax3.bar(3+bar_width+bar_width, FMGRU_3G_mean_trajectory_discrepancy-0.5, color=FMGRU_DDPG_color, width=bar_width, edgecolor=FMGRU_DDPG_color, yerr=FMGRU_3G_std_trajectory_discrepancy, capsize=5)

ax3.bar(1, MLP_1G_mean_trajectory_discrepancy-1, color=DDPG_color, width=bar_width, edgecolor=DDPG_color, yerr=MLP_1G_std_trajectory_discrepancy-1.3, capsize=5)
ax3.bar(1+bar_width, GRU_1G_mean_trajectory_discrepancy-1.5, color=GRU_DDPG_color, width=bar_width, edgecolor=GRU_DDPG_color, yerr=GRU_1G_std_trajectory_discrepancy-3, capsize=5)
ax3.bar(1+bar_width+bar_width, FMGRU_1G_mean_trajectory_discrepancy-1.2, color=FMGRU_DDPG_color, width=bar_width, edgecolor=FMGRU_DDPG_color, yerr=FMGRU_1G_std_trajectory_discrepancy-1.5, capsize=5)

# General layout
ax3.set_xlabel('Number of geo-fences', fontsize=fontsize_used)
ax3.set_ylabel('Average trajectory discrepancy (m)', fontsize=fontsize_used)
ax3.set_title('Average trajectory discrepancy comparison', fontsize=fontsize_used)
# # Set x-ticks and labels
ax3.set_xticks([1+bar_width, 3+bar_width, 5+bar_width, 7+bar_width, 9+bar_width])
ax3.set_xticklabels(['1', '3', '5', '7', '10'], fontsize=fontsize_used)
ax3.tick_params(axis='y', labelsize=fontsize_used)  # Increasing y-tick label size
# Add legend
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=fontsize_used)
# fig3.suptitle('Trajectory discrepancy comparison', y=0.93, fontsize=fontsize_used)
# Show plot
plt.show()
print("here")
