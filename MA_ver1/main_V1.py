# -*- coding: utf-8 -*-
"""
@Time    : 3/1/2023 7:57 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
from parameters_V1 import initialize_parameters

if __name__ == '__main__':
    # initialize parameters
    n_episodes, max_t, eps_start, eps_end, eps_period, eps, env, total_agentNum, \
    agent_grid_obs, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, learning_rate, UPDATE_EVERY = initialize_parameters()
    # create world
    env.create_world(total_agentNum)  # create agents, reset environment


    print('done')

