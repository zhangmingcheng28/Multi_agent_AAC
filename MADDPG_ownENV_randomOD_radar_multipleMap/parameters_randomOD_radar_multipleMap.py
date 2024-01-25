# -*- coding: utf-8 -*-
"""
@Time    : 3/1/2023 7:58 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import numpy as np
from grid_env_generation_newframe_randomOD_radar_multipleMap import env_generation
from env_simulator_randomOD_radar_multipleMap import env_simulator
import pickle

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
    # BUFFER_SIZE = int(1e5)  # replay buffer size
    # BUFFER_SIZE = int(5e4)  # replay buffer size
    BATCH_SIZE = 256  # minibatch size
    GAMMA = 0.90  # discount factor
    TAU = 0.01  # for soft update of target parameters, 0.001, so 99.9% of the weights in the target network is
    learning_rate = 1e-5  # learning rate, previous = 0.0005 or 5e-4, now changed to 0.002
    UPDATE_EVERY = 2  # how often to update the network

    # set boundary
    # xlow = 455
    # xhigh = 680
    # ylow = 255
    # yhigh = 385
    # xlow = 230
    # xhigh = 530
    # ylow = 1000
    # yhigh = 1200
    # xlow = 0
    # xhigh = 1800
    # ylow = 0
    # yhigh = 1300
    # bound = [xlow, xhigh, ylow, yhigh]
    bound = {0: [230, 530, 1000, 1200], 1: [870, 1170, 830, 1030], 2: [100, 400, 500, 700], 3: [455, 680, 255, 385],
             4: [300, 600, 500, 700], 5: [530, 860, 650, 850], 6: [350, 650, 150, 350], 7: [550, 850, 300, 500],
             8: [640, 940, 580, 780], 9: [750, 1050, 150, 350], 10: [880, 1180, 400, 600], 11: [900, 1200, 500, 700],
             12: [930, 1230, 80, 280], 13: [1500, 1800, 300, 500], 14: [280, 580, 0, 200]}

    bound_world_map = {}
    bound_allGridPoly = {}
    cropped_coord_match_actual_coord = {}


    # generate static env from shape file
    shapePath = 'D:\deep_Q_learning\DQN_new_framework\lakesideMap\lakeSide.shp'

    # for bound_idx, bound_val in bound.items():
    #     staticEnv = env_generation(shapePath, bound_val)
    #     bound_world_map[bound_idx] = staticEnv[0]
    #     bound_allGridPoly[bound_idx] = staticEnv[3]
    #     cropped_coord_match_actual_coord[bound_idx] = staticEnv[-1]
    # #
    # with open(r'D:\selected_map\bound_allGridPoly.pickle', 'wb') as handle:
    #     pickle.dump(bound_allGridPoly, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(r'D:\selected_map\bound_world_map.pickle', 'wb') as handle:
    #     pickle.dump(bound_world_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(r'D:\selected_map\whole_map_polygon.pickle', 'wb') as handle:
    #     pickle.dump(staticEnv[1], handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(r'D:\selected_map\cropped_coord_match_actual_coord.pickle', 'wb') as handle:
    #     pickle.dump(cropped_coord_match_actual_coord, handle, protocol=pickle.HIGHEST_PROTOCOL)

    seed = 3407  # this seed is only used for torch manuel.seed
    maxX = 1800
    maxY = 1300
    max_xy = (maxX, maxY)
    # whole_map_polygon = staticEnv[1]
    gridLength = 10
    with open(r'D:\selected_map\bound_allGridPoly.pickle', 'rb') as handle:
        bound_allGridPoly = pickle.load(handle)
    with open(r'D:\selected_map\bound_world_map.pickle', 'rb') as handle:
        bound_world_map = pickle.load(handle)
    with open(r'D:\selected_map\whole_map_polygon.pickle', 'rb') as handle:
        whole_map_polygon = pickle.load(handle)
    with open(r'D:\selected_map\cropped_coord_match_actual_coord.pickle', 'rb') as handle:
        cropped_coord_match_actual_coord = pickle.load(handle)

    # agent config address
    # read the Excel file into a pandas dataframe
    # agentConfig = (r'F:\githubClone\Multi_agent_AAC\MA_ver1\fixedDrone.xlsx')
    # agentConfig = (r'F:\githubClone\Multi_agent_AAC\MA_ver1\fixedDrone_5_adj.xlsx')
    agentConfig = (r'F:\githubClone\Multi_agent_AAC\MA_ver1\fixedDrone_3drones.xlsx')
    # agentConfig = (r'F:\githubClone\Multi_agent_AAC\MA_ver1\fixedDrone_2_drone.xlsx')
    # agentConfig = (r'F:\githubClone\Multi_agent_AAC\MA_ver1\reward_test.xlsx')  # for perform reward testing
    # agentConfig = (r'D:\Multi_agent_AAC\MA_ver1\fixedDrone.xlsx')
    # agentConfig = (r'D:\Multi_agent_AAC\MA_ver1\fixedDrone_3drones.xlsx')
    # agentConfig = (r'D:\Multi_agent_AAC\MA_ver1\fixedDrone_2_drone.xlsx')
    # agentConfig = (r'D:\github_clone\Multi_agent_AAC\MA_ver1\fixedDrone_5_adj.xlsx')
    env = env_simulator(bound_world_map, whole_map_polygon, gridLength, bound, bound_allGridPoly, agentConfig, cropped_coord_match_actual_coord)
    return n_episodes, max_t, eps_start, eps_end, eps_period, eps, env, agent_grid_obs, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, learning_rate, UPDATE_EVERY, seed, max_xy
