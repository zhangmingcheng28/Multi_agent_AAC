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
matplotlib.use('TkAgg')


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

with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\MLP_10G_DP_3_090524_20_20_15.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
mean_deviation_10G = extract_mean_deviation(all_episode_situation)

with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\MADDPG_2nd_jp_090524_20_19_19_episode_situation.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
mean_deviation_10G_DP_1 = extract_mean_deviation(all_episode_situation)

fontsize_used = 14
fig, ax = plt.subplots(figsize=(4.5, 5))
bp0 = ax.boxplot([mean_deviation_10G, mean_deviation_10G_DP_1], labels=['DP_3', 'DP_1'], showmeans=True, meanline=True, patch_artist=True)
# ax.plot(range(len(all_episode_situation)), mean_deviation_10G, label='MLP_mean_deviation_DP_2')
# ax.plot(range(len(all_episode_situation)), mean_deviation_10G_DP_1, label='MLP_mean_deviation_DP_1')

ax.legend(fontsize=fontsize_used, loc='upper left')
ax.set_xlabel('Time steps to goal', fontsize=fontsize_used)
ax.set_ylabel('Deviation at each time steps (m)', fontsize=fontsize_used)

ax.tick_params(axis='y', labelsize=fontsize_used)
ax.tick_params(axis='x', labelsize=fontsize_used)
# ax.set_yticks(np.arange(0, 50, 10), fontsize=fontsize_used)
# ax.set_xticks(np.arange(0, 25, 5), fontsize=fontsize_used)
plt.tight_layout()
plt.show()
