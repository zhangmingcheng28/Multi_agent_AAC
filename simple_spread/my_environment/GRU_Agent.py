from copy import deepcopy
from typing import List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam


class GRU_Agent:
    """single agent in MADDPG"""

    def __init__(self, obs_dim, act_dim, hidden_dim, actor_lr, critic_lr, device):
        """
        create one of the agents in MADDPG
        :param obs_dim: observation dimension of the current agent, i.e. local observation space
        :param act_dim: action dimension of the current agent, i.e. local action space
        :param global_obs_dim: input dimension of the global critic of the current agent, if there are
        3 agents for example, the input for global critic is (obs1, obs2, obs3, act1, act2, act3)
        """
        self.rnn_hidden_dim = hidden_dim
        # the actor output logit of each action
        self.actor = RNN(obs_dim, act_dim, hidden_dim).to(device)
        # critic input all the states and actions
        # self.critic = GRUNetwork(global_obs_dim, 1).to(device)
        self.critic = RNN(obs_dim+act_dim, 1, hidden_dim).to(device)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.target_actor = deepcopy(self.actor).to(device)
        self.target_critic = deepcopy(self.critic).to(device)
        self.device = device

        self.actor_hidden = None
        self.critic_hidden = None

    # def init_hidden(self):
    #     self.actor_hidden = torch.zeros((self.rnn_hidden_dim)).to(self.device)
    #     self.critic_hidden = torch.zeros((self.rnn_hidden_dim)).to(self.device)
    #     return self.actor_hidden


    @staticmethod
    def gumbel_softmax(logits, tau=1.0, eps=1e-20):
        # NOTE that there is a function like this implemented in PyTorch(torch.nn.functional.gumbel_softmax),
        # but as mention in the doc, it may be removed in the future, so i implement it myself
        epsilon = torch.rand_like(logits)
        logits += -torch.log(-torch.log(epsilon + eps) + eps)
        return F.softmax(logits / tau, dim=-1)

    def action(self, obs, hidden_state, *, model_out=False):
        """
        choose action according to given `obs`
        :param model_out: the original output of the actor, i.e. the logits of each action will be
        `gumbel_softmax`ed by default(model_out=False) and only the action will be returned
        if set to True, return the original output of the actor and the action
        """
        # this method is called in the following two cases:
        # a) interact with the environment
        # b) calculate action when update actor, where input(obs) is sampled from replay buffer with size:
        # torch.Size([batch_size, state_dim])

        logits, next_hiddens = self.actor(obs, hidden_state)  # torch.Size([batch_size, action_size])
        action = self.gumbel_softmax(logits)
        if model_out:
            return action, logits,next_hiddens
        return action,next_hiddens

    def target_action(self, obs, hidden_state):
        # when calculate target critic value in MADDPG,
        # we use target actor to get next action given next states,
        # which is sampled from replay buffer with size torch.Size([batch_size, state_dim])
        logits, next_hiddens = self.target_actor(obs, hidden_state)  # torch.Size([batch_size, action_size])
        action = self.gumbel_softmax(logits)
        return action.squeeze(0).detach()

    def critic_value(self, state_list: List[Tensor], act_list: List[Tensor], hidden_list: Tensor):
        x = torch.cat(state_list + act_list, 1)
        h = hidden_list
        q, _ = self.critic(x,h)

        return q.squeeze(1)  # tensor with a given length

    def target_critic_value(self, state_list: List[Tensor], act_list: List[Tensor], hidden_list: Tensor):
        x = torch.cat(state_list + act_list, 1)
        h = hidden_list
        target_q, _ = self.target_critic(x,h)
        return target_q.squeeze(1)  # tensor with a given length

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()




class RNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, n_actions, rnn_hidden_dim):
        super(RNN, self).__init__()

        self.rnn_hidden_dim = rnn_hidden_dim
        self.fc1 = nn.Linear(input_shape, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.rnn_hidden_dim, n_actions)

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


class MLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64, non_linear=nn.ReLU()):
        super(MLPNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, out_dim),
        ).apply(self.init)

    @staticmethod
    def init(m):
        """init parameter of the module"""
        gain = nn.init.calculate_gain('relu')
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.net(x)
