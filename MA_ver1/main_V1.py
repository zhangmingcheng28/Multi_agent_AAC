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
    train_eva = "train"
    random.seed(seed_used)
    # set number of drone in the airspace
    #total_agentNum = random.randint(5, 25)  # max sequence length is 25, meaning in airspace, there are a maximum number of aircraft of 25
    total_agentNum = 5
    # create world
    actor_obs = [6, 20, 6]  # dim host, maximum dim grid, dim other drones
    critic_obs = [6, 20, 6]
    n_actions = 2
    actorNet_lr = learning_rate
    criticNet_lr = learning_rate
    env.create_world(total_agentNum, critic_obs, actor_obs, n_actions, actorNet_lr, criticNet_lr, GAMMA, TAU)  # create agents, reset environment
    # simulation start
    # for i in range(n_episodes):
    combine_state = env.reset_world(show=0)

    # critic network test
    test_critic = env.all_agents[0].criticNet.forward(combine_state, actor_obs)



    for t in range(max_t):
    # get action, no CR is used, output is the velocity
    # actions, noCR = env.get_actions_noCR(combine_state)
    # get action with neural networks
        actions = env.get_actions_NN(combine_state)
    # proceed with the environment step
    #     next_combine_state, reward, drone = env.step(noCR, actions, max_t)

    print('done')

