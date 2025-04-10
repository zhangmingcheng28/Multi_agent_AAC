from Nnetworks_MADDPGv4 import CriticNetwork_0724, ActorNetwork
import torch
from copy import deepcopy
from torch.optim import Adam
from memory_MADDPGv4 import ReplayMemory, Experience
from random_process_MADDPGv4 import OrnsteinUhlenbeckProcess
from torch.autograd import Variable
import os
import torch.nn as nn
import numpy as np
import torch as T
from utils_MADDPGv4 import device
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

        self.actors = [ActorNetwork(actor_dim, dim_act) for _ in range(n_agents)]
        self.critics = [CriticNetwork_0724(critic_dim, n_agents, dim_act) for _ in range(n_agents)]

        self.n_agents = n_agents
        self.n_states = actor_dim
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

    def update_myown(self, i_episode, total_step_count, UPDATE_EVERY):
        self.train_num = i_episode

        # ------------ original -------------
        # if self.train_num <= self.episodes_before_train:
        #     return None, None

        if len(self.memory) <= self.batch_size:
            return None, None

        BoolTensor = torch.cuda.BoolTensor if self.use_cuda else torch.BoolTensor
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        c_loss = []
        a_loss = []

        transitions = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*transitions))
        # cur_state_list1 = [torch.stack([elem[0] for elem in batch.states])[:, i, :] for i in range(self.n_agents)]
        # cur_state_list2 = [torch.stack([elem[1] for elem in batch.states])[:, i, :] for i in range(self.n_agents)]

        action_batch = torch.stack(batch.actions).type(FloatTensor)
        reward_batch = torch.stack(batch.rewards).type(FloatTensor)

        # stack tensors only once
        stacked_elem_0 = torch.stack([elem[0] for elem in batch.states])
        # stacked_elem_1 = torch.stack([elem[1] for elem in batch.states])
        stacked_elem_0_combine = stacked_elem_0.view(self.batch_size, -1)
        # stacked_elem_1_combine = stacked_elem_1.view(self.batch_size, -1)

        # use the stacked tensors
        cur_state_list1 = [stacked_elem_0[:, i, :] for i in range(self.n_agents)]
        # cur_state_list2 = [stacked_elem_1[:, i, :] for i in range(self.n_agents)]
        cur_state_list2 = []
        cur_state_list3 = []

        state_tuple2 = tuple(x[1] for x in batch.states)
        for i in range(self.n_agents):
            all_batch_one_agent = []
            for batch_idx, every_agent_in_batch in enumerate(state_tuple2):
                each_agent_sur_agent = torch.zeros((20, every_agent_in_batch[0].shape[1]))  # 20 is more than the maximum number of observable grids for a single agent
                each_agent_sur_agent[-len(every_agent_in_batch[i]):, :] = every_agent_in_batch[i]  # loading data from the back
                all_batch_one_agent.append(each_agent_sur_agent)
            cur_state_list2.append(torch.stack(all_batch_one_agent))

        state_tuple3 = tuple(x[2] for x in batch.states)
        for i in range(self.n_agents):
            all_batch_one_agent = []
            for batch_idx, every_agent in enumerate(state_tuple3):
                each_agent_sur_agent = torch.zeros((len(every_agent) - 1, every_agent[0].shape[1]))
                each_agent_sur_agent[-len(every_agent[i]):, :] = every_agent[i]
                all_batch_one_agent.append(each_agent_sur_agent)
            cur_state_list3.append(torch.stack(all_batch_one_agent))

        non_final_mask = BoolTensor(list(map(lambda s: True not in s, batch.dones)))   # create a boolean tensor, that has same length as the "batch.next_states", if an element is batch.next_state is not "None" then assign a True value, False otherwise.
        non_final_next_states1_pre = [s_[0] for s_idx, s_ in enumerate(batch.next_states) if non_final_mask[s_idx]]
        non_final_next_states1_combine = torch.stack(non_final_next_states1_pre).view(len(non_final_next_states1_pre), -1)
        non_final_next_states1 = [torch.stack([tensor[i] for tensor in non_final_next_states1_pre], dim=0) for i in range(5)]

        # non_final_next_states2_pre = [s_[1] for s_idx, s_ in enumerate(batch.next_states) if non_final_mask[s_idx]]
        # non_final_next_states2_combine = torch.stack(non_final_next_states2_pre).view(len(non_final_next_states2_pre), -1)
        # non_final_next_states2 = [torch.stack([tensor[i] for tensor in non_final_next_states2_pre], dim=0) for i in range(5)]

        next_state_list2 = [x_[1] for x_idx, x_ in enumerate(batch.next_states) if non_final_mask[x_idx]]
        non_final_next_states2 = []
        for i in range(self.n_agents):
            all_batch_one_agent = []
            for batch_idx, every_agent in enumerate(next_state_list2):
                each_agent_sur_agent = torch.zeros((20, every_agent[0].shape[1]))
                each_agent_sur_agent[-len(every_agent[i]):, :] = every_agent[i]
                all_batch_one_agent.append(each_agent_sur_agent)
            non_final_next_states2.append(torch.stack(all_batch_one_agent))

        next_state_list3 = [x_[2] for x_idx, x_ in enumerate(batch.next_states) if non_final_mask[x_idx]]
        non_final_next_states3 = []
        for i in range(self.n_agents):
            all_batch_one_agent = []
            for batch_idx, every_agent in enumerate(next_state_list3):
                each_agent_sur_agent = torch.zeros((len(every_agent) - 1, every_agent[0].shape[1]))
                each_agent_sur_agent[-len(every_agent[i]):, :] = every_agent[i]
                all_batch_one_agent.append(each_agent_sur_agent)
            non_final_next_states3.append(torch.stack(all_batch_one_agent))

        for agent in range(self.n_agents):
            # --------- original---------
            # non_final_mask = BoolTensor(list(map(lambda s: s is not None, batch.next_states)))  # create a boolean tensor, that has same length as the "batch.next_states", if an element is batch.next_state is not "None" then assign a True value, False otherwise.

            whole_state = [cur_state_list1, cur_state_list2, cur_state_list3]
            # whole_state = [stacked_elem_0_combine, stacked_elem_1_combine]

            non_final_next_states_actorin = [non_final_next_states1, non_final_next_states2, non_final_next_states3]
            # non_final_next_states_criticin = [non_final_next_states1_combine, non_final_next_states2_combine]
            non_final_next_states_criticin = [non_final_next_states1, non_final_next_states2, non_final_next_states3]

            # configured for target Q

            whole_action = action_batch.view(self.batch_size, -1)

            non_final_next_actions = [self.actors_target[i]([non_final_next_states_actorin[0][i], non_final_next_states_actorin[1][i], non_final_next_states_actorin[2][i]]) for i in range(self.n_agents)]  # non_final_next_states[:, i,:]

            non_final_next_actions = torch.stack(non_final_next_actions)
            non_final_next_actions = (non_final_next_actions.transpose(0, 1).contiguous())  # using () at outer most will leads to creation of a new tensor, (batch_size X agentNo X action_dim)

            # get current Q-estimate, using agent's critic network
            current_Q = self.critics[agent](whole_state, whole_action)
            with T.no_grad():
                target_Q = torch.zeros(self.batch_size).type(FloatTensor)
                target_Q[non_final_mask] = self.critics_target[agent](non_final_next_states_criticin, non_final_next_actions.view(-1,self.n_agents * self.n_actions)).squeeze()  # .view(-1, self.n_agents * self.n_actions)
                # target_Q = (target_Q.unsqueeze(1) * self.GAMMA) + (reward_batch[:, agent].unsqueeze(1)*0.1)  # + reward_sum.unsqueeze(1) * 0.1
                target_Q = (target_Q.unsqueeze(1) * self.GAMMA) + (reward_batch[:, agent].unsqueeze(1))  # + reward_sum.unsqueeze(1) * 0.1

            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            self.critic_optimizer[agent].zero_grad()
            loss_Q.backward(retain_graph=True)
            # torch.nn.utils.clip_grad_norm_(self.critics[agent].parameters(), 1)
            self.critic_optimizer[agent].step()

            action_i = self.actors[agent]([cur_state_list1[agent], cur_state_list2[agent], cur_state_list3[agent]])
            ac = action_batch.clone()

            ac[:, agent, :] = action_i  # replace the actor from self.actors[agent] into action batch
            whole_action_action_replaced = ac.view(self.batch_size, -1)

            actor_loss = -self.critics[agent](whole_state, whole_action_action_replaced).mean()
            self.actor_optimizer[agent].zero_grad()
            actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.actors[agent].parameters(), 1)
            self.actor_optimizer[agent].step()

            c_loss.append(loss_Q)
            a_loss.append(actor_loss)

        # original
        if total_step_count % UPDATE_EVERY == 0:  # evert "UPDATE_EVERY" step, do a soft update
            for i in range(self.n_agents):
                print("all agents NN update at total step {}".format(total_step_count))
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)
        # if total_step_count % UPDATE_EVERY == 0:  # evert 100 step, do a soft update
        #     print("all agents NN update at total step {}".format(total_step_count))
        #     for i in range(self.n_agents):
        #         soft_update(self.critics_target[i], self.critics[i], self.tau)
        #         soft_update(self.actors_target[i], self.actors[i], self.tau)

        return c_loss, a_loss

    def choose_action(self, state, episode, noisy=True):
        obs = torch.from_numpy(np.stack(state[0])).float().to(device)
        # obs_grid = torch.from_numpy(np.stack(state[1])).float().to(device)
        all_obs_surGrid = []
        for each_agent_grid in state[1]:
            all_obs_surGrid.append(torch.from_numpy(each_agent_grid).float())
        all_obs_surAgent = []
        for each_agent_sur in state[2]:
            each_obs_surAgent = np.squeeze(np.array(each_agent_sur), axis=1)
            all_obs_surAgent.append(torch.from_numpy(each_obs_surAgent).float())

        # obs_surAgent = torch.from_numpy(np.stack(state[2])).float().to(device)

        actions = torch.zeros(self.n_agents, self.n_actions)
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        for i in range(self.n_agents):
            sb = obs[i].detach()
            sb_grid = all_obs_surGrid[i].detach()
            sb_surAgent = all_obs_surAgent[i].detach()
            act = self.actors[i]([sb.unsqueeze(0), sb_grid.unsqueeze(0), sb_surAgent.unsqueeze(0)]).squeeze()
            if noisy:
                act += torch.from_numpy(np.random.randn(2) * self.var[i]).type(FloatTensor)
                self.var[i] = self.get_scaling_factor(episode, 12000)  # self.var[i] will decrease as the episode increase

                # if self.episode_done > self.episodes_before_train and \
                #         self.var[i] > 0.05:
                #     self.var[i] *= 0.999998
            act = torch.clamp(act, -1.0, 1.0)

            actions[i, :] = act
        self.steps_done += 1
        return actions.data.cpu().numpy()

    def get_scaling_factor(self, episode, drop_point, start_scale=0.5, end_scale=0.03):
        if episode <= drop_point:
            slope = (end_scale - start_scale) / drop_point
            return slope * episode + start_scale
        else:
            return end_scale