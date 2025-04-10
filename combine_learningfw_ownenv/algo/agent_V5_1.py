# -*- coding: utf-8 -*-
"""
@Time    : 3/3/2023 10:10 AM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import numpy as np
import torch as T
from Nnetworks_V5_1 import CriticNetwork
from Nnetworks_V5_1 import ActorNetwork
from Utilities_V5_1 import padding_list


class Agent:
    def __init__(self, actor_obs, critic_obs, n_actions, agent_idx, totalAgent, actorNet_lr, criticNet_lr, gamma, tau, max_nei_num, maxSPD):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions  # this n_actions is the dimension of the action space
        self.agent_name = 'agent_%s' % agent_idx
        self.max_nei = max_nei_num
        #self.agent_size = 1.5  # meter in radius
        self.agent_grid_obs = None
        self.max_grid_obs_dim = actor_obs[1]  # The 2nd element is the maximum grid observation dimension
        self.actorNet = ActorNetwork(actorNet_lr, actor_obs, n_actions, max_nei_num, name=self.agent_name+'_actorNet')
        self.target_actorNet = ActorNetwork(actorNet_lr, actor_obs, n_actions, max_nei_num, name=self.agent_name+'_target_actorNet')
        self.criticNet = CriticNetwork(criticNet_lr, critic_obs, totalAgent, n_actions, name=self.agent_name+'_criticNet')
        self.target_criticNet = CriticNetwork(criticNet_lr, critic_obs, totalAgent, n_actions, name=self.agent_name+'_target_criticNet')

        # state information
        self.pos = None
        self.ini_pos = None
        self.pre_pos = None
        self.vel = None
        self.pre_vel = None
        self.maxSpeed = maxSPD
        self.goal = None
        self.heading = None
        self.detectionRange = 30  # in meters, this is the in diameter
        self.protectiveBound = 2.5  # diameter is 2.5*2, this is radius
        # a dictionary, key is the agent idx, value is the array of 1x6,
        # which correspond to the observation vector of that neighbor
        self.pre_surroundingNeighbor = {}
        self.surroundingNeighbor = {}
        self.observableSpace = []
        self.target_update_step = None
        self.update_count = 0

    def choose_actions(self, observation):
        actions = None
        ownObs = T.tensor(observation[0].reshape(1, -1), dtype=T.float).to(self.actorNet.device)
        # padding actions
        padded_gridObs = padding_list(self.max_grid_obs_dim, observation[1])
        # tobePad_gridObs = list(np.zeros(self.max_grid_obs_dim-len(observation[1]), dtype=int))
        # padded_gridObs = observation[1]+tobePad_gridObs
        onwGridObs = T.tensor([padded_gridObs], dtype=T.float).to(self.actorNet.device)  # Vector in the form of 1xm

        if len(observation[2]) == 0:
            zero_tensor = T.zeros((self.max_nei, ownObs.shape[1])).unsqueeze(0)  # 1x6 zero vector
            # when actor is picking an action, we only use actor's own observation + own grid_observation
            # + neighbors observation, if there is no neighbor detected, we use all zero vector to represent
            actions = self.actorNet.forward([ownObs, onwGridObs, zero_tensor])
        else:
            # handle n x 6
            # neigh_arr = np.zeros((len(self.surroundingNeighbor), ownObs.shape[1]))
            neigh_arr = np.zeros((self.max_nei, ownObs.shape[1]))
            # # ----------------------------------------------------------------------- # #
            # # to do: surrounding neighbour arrange in a way nearest neighbor at 1st or last  # #
            # may not need to do, and make it a novelty such that drone's decision do not depends on
            # arranged input vector. But this cannot consider an novelty, unless, we make a comparison of results
            # between arranged input vector and non-arranged input vectors
            # # ------------------------------------------------------------------------# #
            for neigh_obs_idx, dict_keys in enumerate(self.surroundingNeighbor):  # loop through the dictionary in order, top first
                neigh_arr[neigh_obs_idx, :] = self.surroundingNeighbor[dict_keys]
            neigh_Obs = T.from_numpy(neigh_arr).float().unsqueeze(0).to(self.actorNet.device)
            actions = self.actorNet.forward([ownObs, onwGridObs, neigh_Obs])
        return actions

    def update_network_parameters(self):
        # update actor target network parameters
        for target_actor_param, actor_param in zip(self.target_actorNet.parameters(), self.actorNet.parameters()):
            target_actor_param.data.copy_(target_actor_param.data * (1.0-self.tau) + actor_param * self.tau)

        # update critic target network parameters
        for target_critic_param, critic_param in zip(self.target_criticNet.parameters(), self.criticNet.parameters()):
            target_critic_param.data.copy_(target_critic_param.data * (1.0-self.tau) + critic_param * self.tau)



