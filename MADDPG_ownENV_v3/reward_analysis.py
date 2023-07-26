# -*- coding: utf-8 -*-
"""
@Time    : 7/26/2023 9:51 AM
@Author  : Thu Ra
@FileName: 
@Description: 
@Package dependency:
"""
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
pre_fix = r'D:\MADDPG_2nd_jp\250723_16_14_40\toplot'
file_path = pre_fix + r'\all_episode_reward.pickle'
with open(file_path, 'rb') as handle:
    combine_reward = pickle.load(handle)

# 1st term is cross-track error term
# 2nd term is goal term
# 3rd term is the ailve penalty
CT_term = []
g_term = []
alive_term = []
for per_ep_r in combine_reward:
    for per_step_r in per_ep_r:
        for per_agent_r in per_step_r:
            if per_agent_r is None:
                # print("None is found in a step of episode, skip it")
                continue
            CT_term.append(per_agent_r[0])
            g_term.append(per_agent_r[1])
            alive_term.append(per_agent_r[2])
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
fig, ax = plt.subplots(1, 1)

# plot1 = plt.plot(CT_term, linestyle='-', label='cross-track-term')
plot2 = plt.plot(g_term, linestyle='--', label='goal-term')
# plot3 = plt.plot(alive_term, linestyle='-.', label='alive-term')

plt.grid()
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.show()
print('done')