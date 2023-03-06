# -*- coding: utf-8 -*-
"""
@Time    : 3/1/2023 7:57 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
from parameters_V1 import initialize_parameters
import random

if __name__ == '__main__':
    # initialize parameters
    n_episodes, max_t, eps_start, eps_end, eps_period, eps, env, \
    agent_grid_obs, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, learning_rate, UPDATE_EVERY, seed_used = initialize_parameters()

    random.seed(seed_used)
    # set number of drone in the airspace
    total_agentNum = random.randint(5, 25)  # max sequence length is 25, meaning in airspace, there are a maximum number of aircraft of 25
    # create world
    env.create_world(total_agentNum, learning_rate, GAMMA, TAU, agent_grid_obs, )  # create agents, reset environment


    print('done')

