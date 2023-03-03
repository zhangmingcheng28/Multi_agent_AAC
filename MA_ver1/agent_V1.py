# -*- coding: utf-8 -*-
"""
@Time    : 3/3/2023 10:10 AM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import torch as T
from Nnetworks import CriticNetwork
from Nnetworks import ActorNetwork


class Agent:
    def __init__(self, actor_dim, critic_dims, n_actions, agent_idx, totalAgent, actorNet_lr, criticNet_lr, gamma, tau):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx
        self.actorNet = ActorNetwork(actorNet_lr, actor_dim, n_actions, name=self.agent_name+'_actorNet')
        self.target_actorNet = ActorNetwork(actorNet_lr, actor_dim, n_actions, name=self.agent_name+'_target_actorNet')
        self.criticNet = CriticNetwork(criticNet_lr, critic_dims, totalAgent, n_actions, name=self.agent_name+'_criticNet')
        self.target_criticNet = CriticNetwork(criticNet_lr, critic_dims, totalAgent, n_actions, name=self.agent_name+'_target_criticNet')
        self.update_network_parameters()

    def choose_actions(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actorNet.device)
        actions = self.actorNet.forward(state)
        return actions

    def update_network_parameters(self):
        # update actor target network parameters
        for target_actor_param, actor_param in zip(self.target_actorNet.parameters(), self.actorNet.parameters()):
            target_actor_param.data.copy_(target_actor_param.data * (1.0-self.tau) + actor_param * self.tau)

        # update critic target network parameters
        for target_critic_param, critic_param in zip(self.target_criticNet.parameters(), self.criticNet.parameters()):
            target_critic_param.data.copy_(target_critic_param.data * (1.0-self.tau) + critic_param * self.tau)
