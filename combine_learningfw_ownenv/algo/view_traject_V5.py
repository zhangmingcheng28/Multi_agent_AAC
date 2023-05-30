# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import csv
import pickle
import random
from Multi_Agent_replaybuffer_V5 import MultiAgentReplayBuffer
import os
from parameters_V5 import initialize_parameters
from Utilities_V5 import sort_polygons, shapelypoly_to_matpoly, \
    extract_individual_obs, map_range, compute_potential_conflict, display_trajectory, action_selection_statistics
matplotlib.use('TkAgg')

n_episodes, max_t, eps_start, eps_end, eps_period, eps, env, \
agent_grid_obs, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, learning_rate, UPDATE_EVERY, seed_used = initialize_parameters()
train_eva = "train"
random.seed(seed_used)
# set number of drone in the airspace
total_agentNum = 5
max_nei_num = 5
# create world
actor_obs = [6 + (total_agentNum - 1) * 2, 20, 6]  # dim host, maximum dim grid, dim other drones
critic_obs = [6 + (total_agentNum - 1) * 2, 20, 6]
n_actions = 2
actorNet_lr = learning_rate
criticNet_lr = learning_rate
# noise parameter ini
largest_Nsigma = 0.15
smallest_Nsigma = 0.01
ini_Nsigma = largest_Nsigma

# create agents, reset environment
env.create_world(total_agentNum, critic_obs, actor_obs, n_actions, actorNet_lr, criticNet_lr, GAMMA, TAU, UPDATE_EVERY,
                 largest_Nsigma, smallest_Nsigma, ini_Nsigma, max_nei_num)

# initialized memory replay
actor_dims = 3  # A list of 3 list, each 1st list has length 3, 2nd has length 20, 3rd has length 6
critic_dims = total_agentNum * actor_dims  # critic is centralized, so we combine dim of all agents
ReplayBuffer = MultiAgentReplayBuffer(BUFFER_SIZE, actor_dims, critic_dims, total_agentNum, n_actions,
                                      batch_size=BATCH_SIZE)
# print("time to initiate is {}".format(time.time()-start_time))
score_history = []
Trajectory_history = []
Trajectory_action_record = []

# get navigate to plot file and load pickle
with open(r'D:\MADDPG_2nd_jp\260523_15_55_57\toplot\all_episode_trajectory.pickle', 'rb') as handle:
    all_trajectory = pickle.load(handle)

with open(r'D:\MADDPG_2nd_jp\270423_15_15_28\toplot\all_episode_action_taken.pickle', 'rb') as handle:
    action_collection = pickle.load(handle)

reward = display_trajectory(env, all_trajectory)
# action_selection_statistics(action_collection)
