# from Nnetworks_MADDPGv3 import CriticNetwork_0724, ActorNetwork
from Nnetworks_MADDPGv3_ss import CriticNetwork, ActorNetwork, Stocha_actor
import torch
from copy import deepcopy
from torch.optim import Adam
from memory_MADDPGv3_ss import ReplayMemory, Experience
from random_process_MADDPGv3_ss import OrnsteinUhlenbeckProcess
from torch.autograd import Variable
import os
import torch.nn as nn
import numpy as np
import torch as T
from utils_MADDPGv3_ss import device
import csv
scale_reward = 0.01


def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)


class MADDPG:
    def __init__(self, actor_dim, critic_dim, dim_act, n_agents, args, cr_lr, ac_lr, gamma, tau):
        self.args = args
        self.mode = args.mode
        self.actors = []
        self.critics = []
        # original
        # self.actors = [Actor(dim_obs, dim_act) for _ in range(n_agents)]
        # self.critics = [Critic(n_agents, dim_obs, dim_act) for _ in range(n_agents)]

        # self.actors = [Stocha_actor(actor_dim, dim_act) for _ in range(n_agents)]  # use stochastic policy
        self.actors = [ActorNetwork(actor_dim, dim_act) for _ in range(n_agents)]  # use deterministic policy
        # self.critics = [CriticNetwork_0724(critic_dim, n_agents, dim_act) for _ in range(n_agents)]
        self.critics = [CriticNetwork(critic_dim, n_agents, dim_act) for _ in range(n_agents)]

        self.n_agents = n_agents
        self.n_actor_dim = actor_dim
        self.n_critic_dim = critic_dim
        self.n_actions = dim_act

        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.memory = ReplayMemory(args.memory_length)
        self.batch_size = args.batch_size
        self.use_cuda = torch.cuda.is_available()
        self.episodes_before_train = args.episode_before_train

        # self.GAMMA = 0.95  # original
        # self.tau = 0.01  # original

        self.GAMMA = gamma
        self.tau = tau

        # self.var = [1.0 for i in range(n_agents)]
        self.var = [0.5 for i in range(n_agents)]

        # original, critic learning rate is 10 times larger compared to actor
        # self.critic_optimizer = [Adam(x.parameters(), lr=0.001) for x in self.critics]
        # self.actor_optimizer = [Adam(x.parameters(), lr=0.0001) for x in self.actors]

        self.critic_optimizer = [Adam(x.parameters(), lr=cr_lr) for x in self.critics]
        self.actor_optimizer = [Adam(x.parameters(), lr=ac_lr) for x in self.actors]

        if self.use_cuda:
            for x in self.actors:
                x.cuda()
            for x in self.critics:
                x.cuda()
            for x in self.actors_target:
                x.cuda()
            for x in self.critics_target:
                x.cuda()

        self.steps_done = 0
        self.episode_done = 0

    def load_model(self, filePath):
        if self.args.model_episode:
            path_flag = True
            # for idx in range(self.n_agents):
            #     path_flag = path_flag \
            #                 and (os.path.exists("trained_model/maddpg/actor["+ str(idx) + "]_"
            #                                     +str(self.args.model_episode)+".pth")) \
            #                 and (os.path.exists("trained_model/maddpg/critic["+ str(idx) + "]_"
            #                                     +str(self.args.model_episode)+".pth"))

            if path_flag:
                print("load model!")
                # for idx in range(self.n_agents):
                #     actor = torch.load("trained_model/maddpg/actor["+ str(idx) + "]_"+str(self.args.model_episode)+".pth")
                #     critic = torch.load("trained_model/maddpg/critic["+ str(idx) + "]_"+str(self.args.model_episode)+".pth")
                #     self.actors[idx].load_state_dict(actor.state_dict())
                #     self.critics[idx].load_state_dict(critic.state_dict())
                for path_idx, path in enumerate(filePath):
                    self.actors[path_idx].load_state_dict(torch.load(path))



        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

    def save_model(self, episode, file_path):
        # if not os.path.exists("./trained_model_myenv/"):
        #     os.mkdir("./trained_model_myenv/")
        # if not os.path.exists("./trained_model/" + str(self.args.algo) + "/"):
        #     # os.mkdir(r"F:\githubClone\MAProj_myversion\algo/trained_model/" + str(self.args.algo))
        #     os.mkdir(r"D:\Multi_agent_AAC\old_framework_test\algo/trained_model/" + str(self.args.algo))
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        for i in range(self.n_agents):
            torch.save(self.actors[i].state_dict(), file_path + '/' +'episode_'+str(episode)+'_'+'agent_'+ str(i) + 'actor_net.pth')

    def update(self, i_episode):

        self.train_num = i_episode
        if self.train_num <= self.episodes_before_train:
            return None, None

        BoolTensor = torch.cuda.BoolTensor if self.use_cuda else torch.BoolTensor
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        c_loss = []
        a_loss = []

        transitions = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*transitions))

        for agent in range(self.n_agents):

            non_final_mask = BoolTensor(list(map(lambda s: s is not None,
                                                 batch.next_states)))  # create a boolean tensor, that has same length as the "batch.next_states", if an element is batch.next_state is not "None" then assign a True value, False otherwise.
            # state_batch: batch_size x n_agents x dim_obs
            state_batch = torch.stack(batch.states).type(FloatTensor)
            action_batch = torch.stack(batch.actions).type(FloatTensor)
            reward_batch = torch.stack(batch.rewards).type(FloatTensor)
            non_final_next_states = torch.stack([s for s in batch.next_states if s is not None]).type(FloatTensor)  # create a new tensor, but exclude the None values in the old tensor which is "batch.next_states"
            whole_state = state_batch.view(self.batch_size, -1)
            whole_action = action_batch.view(self.batch_size, -1)

            self.actor_optimizer[agent].zero_grad()
            self.critic_optimizer[agent].zero_grad()
            self.actors[agent].zero_grad()
            self.critics[agent].zero_grad()

            current_Q = self.critics[agent](whole_state, whole_action)
            non_final_next_actions = [self.actors_target[i](non_final_next_states[:, i,:]) for i in range(self.n_agents)]
            non_final_next_actions = torch.stack(non_final_next_actions)
            non_final_next_actions = (non_final_next_actions.transpose(0,1).contiguous())  # using () at outer most will leads to creation of a new tensor, (batch_size X agentNo X action_dim)

            target_Q = torch.zeros(self.batch_size).type(FloatTensor)
            target_Q[non_final_mask] = self.critics_target[agent](
                non_final_next_states.view(-1, self.n_agents * self.n_states), # .view(-1, self.n_agents * self.n_states)
                non_final_next_actions.view(-1, self.n_agents * self.n_actions)).squeeze() # .view(-1, self.n_agents * self.n_actions)

            # scale_reward: to scale reward in Q functions
            reward_sum = sum([reward_batch[:,agent_idx] for agent_idx in range(self.n_agents)])
            target_Q = (target_Q.unsqueeze(1) * self.GAMMA) + (reward_batch[:, agent].unsqueeze(1))
            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            loss_Q.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[agent].parameters(), 1)
            self.critic_optimizer[agent].step()

            self.actor_optimizer[agent].zero_grad()
            self.critic_optimizer[agent].zero_grad()
            self.actors[agent].zero_grad()
            self.critics[agent].zero_grad()

            state_i = state_batch[:, agent, :]
            action_i = self.actors[agent](state_i)
            ac = action_batch.clone()
            # ac = action_batch
            ac[:, agent, :] = action_i  # replacing every single element in "ac[:, agent, :]"
            whole_action = ac.view(self.batch_size, -1)

            actor_loss = -self.critics[agent](whole_state, whole_action).mean()
            # actor_loss += (action_i ** 2).mean() * 1e-3
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[agent].parameters(), 1)
            torch.nn.utils.clip_grad_norm_(self.critics[agent].parameters(), 1)
            self.actor_optimizer[agent].step()
            # self.critic_optimizer[agent].step()
            c_loss.append(loss_Q)
            a_loss.append(actor_loss)

        if self.train_num % 100 == 0:
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)

        return c_loss, a_loss

    def update_myown(self, i_episode, total_step_count, UPDATE_EVERY, wandb=None):
        self.train_num = i_episode

        if len(self.memory) <= self.batch_size:
            return None, None

        BoolTensor = torch.cuda.BoolTensor if self.use_cuda else torch.BoolTensor
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        c_loss = []
        a_loss = []

        transitions = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*transitions))

        action_batch = torch.stack(batch.actions).type(FloatTensor)
        reward_batch = torch.stack(batch.rewards).type(FloatTensor)
        state_batch = torch.stack(batch.states).type(FloatTensor)
        whole_state = state_batch.view(self.batch_size, -1)  # current state
        whole_action = action_batch.view(self.batch_size, -1)  # current action from experience replay
        non_final_next_states = torch.stack([s for s in batch.next_states if s is not None]).type(FloatTensor)
        non_final_next_actions = [self.actors_target[i](non_final_next_states[:, i, :]) for i in range(self.n_agents)]
        non_final_next_actions = torch.stack(non_final_next_actions)
        non_final_next_actions = (non_final_next_actions.transpose(0, 1).contiguous())  # using () at outer most will leads to creation of a new tensor, (batch_size X agentNo X action_dim)

        # non_final_mask = BoolTensor(list(map(lambda s: True not in s, batch.dones)))   # create a boolean tensor, that has same length as the "batch.next_states", if an element is batch.next_state is not "None" then assign a True value, False otherwise.
        non_final_mask = BoolTensor(list(map(lambda s: s is not None, batch.next_states)))  # create a boolean tensor, that has same length as the "batch.next_states", if an element is batch.next_state is not "None" then assign a True value, False otherwise.

        for agent in range(self.n_agents):
            # get current Q-estimate, using agent's critic network
            current_Q = self.critics[agent](whole_state, whole_action)
            # with T.no_grad():
            target_Q = torch.zeros(self.batch_size).type(FloatTensor)
            target_Q[non_final_mask] = self.critics_target[agent](non_final_next_states.view(-1, self.n_agents * self.n_actor_dim),
                                                                  non_final_next_actions.view(-1,self.n_agents * self.n_actions)).squeeze()  # .view(-1, self.n_agents * self.n_actions)
            target_Q = (target_Q.unsqueeze(1) * self.GAMMA) + (reward_batch[:, agent].unsqueeze(1))

            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            self.critic_optimizer[agent].zero_grad()
            loss_Q.backward(retain_graph=True)

            self.critic_optimizer[agent].step()

            action_i = self.actors[agent](state_batch[:, agent, :])
            ac = action_batch.clone()

            ac[:, agent, :] = action_i  # replace the actor from self.actors[agent] into action batch
            whole_action_action_replaced = ac.view(self.batch_size, -1)

            actor_loss = -self.critics[agent](whole_state, whole_action_action_replaced).mean()
            self.actor_optimizer[agent].zero_grad()
            actor_loss.backward()

            self.actor_optimizer[agent].step()

            c_loss.append(loss_Q)
            a_loss.append(actor_loss)

        # original
        if total_step_count % UPDATE_EVERY == 0:  # evert "UPDATE_EVERY" step, do a soft update
            for i in range(self.n_agents):
                print("all agents NN update at total step {}".format(total_step_count))
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)

        return c_loss, a_loss

    def has_gradients(self, model, wandb=None):
        for name, param in model.named_parameters():
            if param.grad is None:
                print(f"No gradient for {name}")
            else:
                # print(f"Gradient for {name} is {param.grad.norm()}")
                wandb.log({name: float(param.grad.norm())})

    def choose_action(self, state, episode, step, eps_end, noise_start_level, noisy=True):
        obs = torch.from_numpy(np.stack(state)).float().to(device)

        # obs = torch.from_numpy(np.stack(state[0])).float().to(device)
        # obs_grid = torch.from_numpy(np.stack(state[1])).float().to(device)
        # noise_value = np.zeros(2)
        # all_obs_surAgent = []
        # for each_agent_sur in state[2]:
        #     try:
        #         each_obs_surAgent = np.squeeze(np.array(each_agent_sur), axis=1)
        #         all_obs_surAgent.append(torch.from_numpy(each_obs_surAgent).float().to(device))
        #     except:
        #         print("pause and check")

        # obs_surAgent = torch.from_numpy(np.stack(state[2])).float().to(device)

        actions = torch.zeros(self.n_agents, self.n_actions)
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        # this for loop used to decrease noise level for all agents before taking any action
        for i in range(self.n_agents):
            self.var[i] = self.get_custom_linear_scaling_factor(episode, eps_end, noise_start_level)

        for i in range(self.n_agents):
            # sb = obs[i].detach()
            # sb_grid = obs_grid[i].detach()
            # sb_surAgent = all_obs_surAgent[i].detach()
            sb = obs[i]
            # sb_grid = obs_grid[i]
            # sb_surAgent = all_obs_surAgent[i]
            # act = self.actors[i]([sb.unsqueeze(0), sb_grid.unsqueeze(0), sb_surAgent.unsqueeze(0)]).squeeze()
            # act = self.actors[i]([sb.unsqueeze(0), sb_surAgent.unsqueeze(0)]).squeeze()
            act = self.actors[i](sb.unsqueeze(0)).squeeze()
            if noisy:
                noise_value = np.random.randn(2) * self.var[i]
                act += torch.from_numpy(noise_value).type(FloatTensor)
                # print("Episode {}, agent {}, noise level is {}".format(episode, i, self.var[i]))

            # act = torch.clamp(act, -1.0, 1.0)  # when using stochastic policy, we are not require to clamp again.

            actions[i, :] = act
        self.steps_done += 1
        return actions.data.cpu().numpy(), noise_value

    def get_custom_linear_scaling_factor(self, episode, eps_end, start_scale=1, end_scale=0.03):
        # Calculate the slope of the linear decrease only up to eps_end
        if episode <= eps_end:
            slope = (end_scale - start_scale) / (eps_end - 1)
            current_scale = start_scale + slope * (episode - 1)
        else:
            current_scale = end_scale
        return current_scale