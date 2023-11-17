# -*- coding: utf-8 -*-
"""
@Time    : 11/16/2023 9:42 AM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
pre_fix = r'D:\MADDPG_2nd_jp\161123_09_51_46\toplot'
file_path = pre_fix + r'\all_episode_noise.pickle'
with open(file_path, 'rb') as handle:
    combine_noise = pickle.load(handle)

# x-acc noise
x_acc_noise = []
y_acc_noise = []
for each_eps in combine_noise:
    for each_step in each_eps:
        x_acc_noise.append(each_step[0])
        y_acc_noise.append(each_step[1])

plot1 = plt.plot(x_acc_noise, label='x_noise')
plot2 = plt.plot(y_acc_noise, label='y_noise')

plt.grid(linestyle='-.')
plt.xlabel('Episode')
plt.ylabel('Ave reward')
plt.legend()
plt.show()
print('end')