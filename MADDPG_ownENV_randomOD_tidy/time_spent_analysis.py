# -*- coding: utf-8 -*-
"""
@Time    : 11/27/2023 2:29 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('TkAgg')
pre_fix = r'D:\MADDPG_2nd_jp\271123_14_33_40\toplot'
file_path = pre_fix + r'\all_episode_time.pickle'
with open(file_path, 'rb') as handle:
    combine_time = pickle.load(handle)

plt.figure(1)  # each episode reset time consumed in milliseconds
reset_milliseconds = []
for ea_eps_reset in combine_time:
    reset_milliseconds.append(ea_eps_reset[0])
plot1 = plt.plot(reset_milliseconds, label='episode reset time')
plt.show()

plt.figure(1)  # each episode reset time consumed in milliseconds


print("hi")