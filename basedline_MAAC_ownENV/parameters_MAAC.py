# -*- coding: utf-8 -*-
"""
@Time    : 3/1/2023 7:58 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import numpy as np
from grid_env_generation_newframe_MAAC import env_generation
# from env_simulator_MAAC import env_simulator
from env_simulator_MAAC_randomOD import env_simulator


def initialize_parameters():
    n_episodes = 12000
    max_t = 45
    eps_start = 1.0
    eps_end = 0.05  # The minimum number of epsilon
    #eps_decay = 0.99992
    eps_period = 8500  # The number of steps needed for the epsilon to drop until the minimum number of the "eps_end"
    eps = eps_start  # initialize epsilon
    agent_grid_obs = np.zeros((7, 7))
    agent_obs_dim = 6  # Vx, Vy, delta_Gx, delta_Gy, Acc_x, Acc_y
    BUFFER_SIZE = int(1e6)  # replay buffer size
    BATCH_SIZE = 256  # minibatch size
    GAMMA = 0.99  # discount factor
    TAU = 1e-3  # for soft update of target parameters, 0.001, so 99.9% of the weights in the target network is
    learning_rate = 1e-5  # learning rate, previous = 0.0005 or 5e-4, now changed to 0.002
    UPDATE_EVERY = 5  # how often to update the network
    # set boundary
    xlow = 455
    xhigh = 680
    ylow = 255
    yhigh = 385
    bound = [xlow, xhigh, ylow, yhigh]
    # generate static env from shape file
    shapePath = 'D:\deep_Q_learning\DQN_new_framework\lakesideMap\lakeSide.shp'
    # shapePath = 'F:\githubClone\deep_Q_learning\DQN_new_framework\lakesideMap\lakeSide.shp'
    staticEnv = env_generation(shapePath, bound)  # it is a tuple of 4 element, 1st is the 2D binary array of the filled map, 2nd is the list of all buildings expressed as polygons, 3rd is the gird length, last is list of square grids in the map that has overlapped with the building polygons.
    seed = 3407  # this seed is only used for torch manuel.seed

    # agent config address
    # read the Excel file into a pandas dataframe
    # agentConfig = (r'F:\githubClone\Multi_agent_AAC\MA_ver1\fixedDrone.xlsx')
    # agentConfig = (r'F:\githubClone\Multi_agent_AAC\MA_ver1\fixedDrone_5_adj.xlsx')
    # agentConfig = (r'F:\githubClone\Multi_agent_AAC\MA_ver1\fixedDrone_2_drone.xlsx')
    agentConfig = (r'D:\Multi_agent_AAC\MA_ver1\fixedDrone.xlsx')
    # agentConfig = (r'D:\Multi_agent_AAC\MA_ver1\fixedDrone_2_drone.xlsx')
    env = env_simulator(staticEnv[0], staticEnv[1], staticEnv[2], bound, staticEnv[3], agentConfig)
    return n_episodes, max_t, eps_start, eps_end, eps_period, eps, env, agent_grid_obs, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, learning_rate, UPDATE_EVERY, seed, staticEnv[-1]
