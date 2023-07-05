import torch
import torch.nn.functional as F
import numpy as np
import os
from torch.optim import Adam
from utils_MAAC.misc import soft_update, hard_update, enable_gradients, disable_gradients, device
from utils_MAAC.agents import AttentionAgent
from utils_MAAC.critics import AttentionCritic

MSELoss = torch.nn.MSELoss()

class AttentionSAC(object):
    """
    Wrapper class for SAC agents with central attention critic in multi-agent
    task
    """
    def __init__(self, agent_init_params, sa_size,
                 gamma=0.95, tau=0.01, pi_lr=0.01, q_lr=0.01,
                 reward_scale=10.,
                 pol_hidden_dim=128,
                 critic_input_dim=None,
                 critic_hidden_dim=128, attend_heads=4,
                 **kwargs):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
            sa_size (list of (int, int)): Size of state and action space for
                                          each agent
            gamma (float): Discount factor
            tau (float): Target update rate
            pi_lr (float): Learning rate for policy
            q_lr (float): Learning rate for critic
            reward_scale (float): Scaling for reward (has effect of optimal
                                  policy entropy)
            hidden_dim (int): Number of hidden dimensions for networks
        """
        self.nagents = len(sa_size)

        self.agents = [AttentionAgent(lr=pi_lr,
                                      hidden_dim=pol_hidden_dim,
                                      **params)
                         for params in agent_init_params]
        self.critic = AttentionCritic(sa_size, critic_input_dim, hidden_dim=critic_hidden_dim,
                                      attend_heads=attend_heads)
        self.target_critic = AttentionCritic(sa_size, critic_input_dim, hidden_dim=critic_hidden_dim,
                                             attend_heads=attend_heads)
        hard_update(self.target_critic, self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=q_lr,
                                     weight_decay=1e-3)
        self.var = [1.0 for _ in range(len(sa_size))]
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.reward_scale = reward_scale
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def step(self, state, episode, explore=False):
        obs = torch.from_numpy(np.stack(state[0])).float().to(device)
        obs_grid = torch.from_numpy(np.stack(state[1])).float().to(device)
        all_obs_surAgent = []
        for each_agent_sur in state[2]:
            each_obs_surAgent = np.squeeze(np.array(each_agent_sur), axis=1)
            all_obs_surAgent.append(torch.from_numpy(each_obs_surAgent).float())

        actions = torch.zeros(len(self.agents), self.agent_init_params[0]['num_out_pol'])
        FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        for i in range(len(self.agents)):
            sb = obs[i].detach()
            sb_grid = obs_grid[i].detach()
            sb_surAgent = all_obs_surAgent[i].detach()
            act = self.agents[i].step([sb.unsqueeze(0), sb_grid.unsqueeze(0), sb_surAgent.unsqueeze(0)], explore=explore)
            # if explore:
            #     act += torch.from_numpy(np.random.randn(2) * self.var[i]).type(FloatTensor)
            #     self.var[i] = self.get_scaling_factor(episode, 12000)  # self.var[i] will decrease as the episode increase
            #
            # act = torch.clamp(act, -1.0, 1.0)
            #
            actions[i, :] = act
        return actions.data.cpu().numpy()

    def get_scaling_factor(self, episode, drop_point, start_scale=1, end_scale=0.03):
        if episode <= drop_point:
            slope = (end_scale - start_scale) / drop_point
            return slope * episode + start_scale
        else:
            return end_scale

    def update_critic(self, sample, soft=True, **kwargs):
        """
        Update central critic for all agents
        """
        obs, acs, rews, next_obs, dones = sample
        rews = list(rews.transpose(0,1))
        dones = list(dones.transpose(0,1))
        all_probs = []
        all_log_pis = []
        all_pol_regs = []

        # Q loss
        next_acs = []
        next_log_pis = []
        for pi_idx, pi in enumerate(self.target_policies):
            curr_next_ac, curr_next_log_pi = pi([next_obs[0][pi_idx], next_obs[1][pi_idx], next_obs[2][pi_idx]],
                                                       return_log_pi=True)
            next_acs.append(curr_next_ac)
            next_log_pis.append(curr_next_log_pi)

        trgt_critic_in = (next_obs, next_acs)
        next_qs = self.target_critic(trgt_critic_in)  # input is a list, length is total number of agents. Each element holds each agent's next state and action with batch information.
        acs = list(acs.transpose(0, 1))
        critic_in = (obs, acs)
        critic_rets = self.critic(critic_in, regularize=True, niter=self.niter)
        q_loss = 0
        for a_i, nq, log_pi, (pq, regs) in zip(range(5), next_qs, next_log_pis, critic_rets):
            target_q = (rews[a_i].view(-1, 1) + self.gamma * nq * (1 - dones[a_i].view(-1, 1)))
            if soft:
                target_q -= log_pi / self.reward_scale  # this reward_scale is set to 100
            q_loss += MSELoss(pq, target_q.detach())  # summing the MSE loss across all agents
            for reg in regs:
                q_loss += reg  # regularizing attention
        q_loss.backward()
        self.critic.scale_shared_grads()
        # grad_norm = torch.nn.utils.clip_grad_norm(
        #     self.critic.parameters(), 10 * self.nagents)  # originally used for logger
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

        self.niter += 1

    def update_policies(self, sample, soft=True, **kwargs):
        obs, acs, rews, next_obs, dones = sample
        samp_acs = []
        all_probs = []
        all_log_pis = []
        all_pol_regs = []
        for pi_idx, pi in enumerate(self.policies):
            curr_ac, probs, curr_log_pi, pol_regs = pi([obs[0][pi_idx], obs[1][pi_idx], obs[2][pi_idx]],
                                                       return_all_probs=True, return_log_pi=True, regularize=True)
            samp_acs.append(curr_ac)
            all_probs.append(probs)
            all_log_pis.append(curr_log_pi)
            all_pol_regs.append(pol_regs)

        # critic_in = list(zip(obs, samp_acs))
        critic_in = (obs, samp_acs)
        critic_rets = self.critic(critic_in, return_all_q=True)
        for a_i, probs, log_pi, pol_regs, (q, all_q) in zip(range(5), all_probs,
                                                            all_log_pis, all_pol_regs,
                                                            critic_rets):
            curr_agent = self.agents[a_i]
            v = (all_q * probs).sum(dim=1, keepdim=True)
            pol_target = q - v
            if soft:
                pol_loss = (log_pi * (log_pi / self.reward_scale - pol_target).detach()).mean()
            else:
                pol_loss = (log_pi * (-pol_target).detach()).mean()
            for reg in pol_regs:
                pol_loss += 1e-3 * reg  # policy regularization
            # don't want critic to accumulate gradients from policy loss
            disable_gradients(self.critic)
            pol_loss.backward()
            enable_gradients(self.critic)

            grad_norm = torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), 0.5)
            curr_agent.policy_optimizer.step()
            curr_agent.policy_optimizer.zero_grad()


    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        soft_update(self.target_critic, self.critic, self.tau)
        for a in self.agents:
            soft_update(a.target_policy, a.policy, self.tau)

    def prep_training(self, device='gpu'):
        self.critic.train()
        self.target_critic.train()
        for a in self.agents:
            a.policy.train()
            a.target_policy.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            self.critic = fn(self.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            self.target_critic = fn(self.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents],
                     'critic_params': {'critic': self.critic.state_dict(),
                                       'target_critic': self.target_critic.state_dict(),
                                       'critic_optimizer': self.critic_optimizer.state_dict()}}
        torch.save(save_dict, filename)

    def save_model(self, episode, file_path, n_agents):
        # if not os.path.exists("./trained_model_myenv/"):
        #     os.mkdir("./trained_model_myenv/")
        # if not os.path.exists("./trained_model/" + str(self.args.algo) + "/"):
        #     # os.mkdir(r"F:\githubClone\MAProj_myversion\algo/trained_model/" + str(self.args.algo))
        #     os.mkdir(r"D:\Multi_agent_AAC\old_framework_test\algo/trained_model/" + str(self.args.algo))
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        for i in range(n_agents):
            torch.save(self.policies[i].state_dict(), file_path + '/' +'episode_'+str(episode)+'_'+'agent_'+ str(i) + 'actor_net')


    @classmethod
    def init_from_env(cls, env, critic_dim, gamma=0.95, tau=0.01,
                      pi_lr=0.01, q_lr=0.01,
                      reward_scale=10.,
                      pol_hidden_dim=128, critic_hidden_dim=128, attend_heads=4, actor_dim=0,
                      **kwargs):
        """
        Instantiate instance of this class from multi-agent environment

        env: Multi-agent Gym environment
        gamma: discount factor
        tau: rate of update for target networks
        lr: learning rate for networks
        hidden_dim: number of hidden dimensions for networks
        """
        agent_init_params = []
        sa_size = []
        for i in range(len(env.all_agents)):
            sa_size.append((env.env_combined_obs_space, env.env_combined_action_space))
            agent_init_params.append({'num_in_pol': actor_dim,
                                      'num_out_pol': env.env_combined_action_space})
            sa_size.append((env.env_combined_obs_space, env.env_combined_action_space))

        init_dict = {'gamma': gamma, 'tau': tau,
                     'pi_lr': pi_lr, 'q_lr': q_lr,
                     'reward_scale': reward_scale,
                     'pol_hidden_dim': pol_hidden_dim,
                     'critic_input_dim': critic_dim,
                     'critic_hidden_dim': critic_hidden_dim,
                     'attend_heads': attend_heads,
                     'agent_init_params': agent_init_params,
                     'sa_size': sa_size}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename, load_critic=False):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)

        if load_critic:
            critic_params = save_dict['critic_params']
            instance.critic.load_state_dict(critic_params['critic'])
            instance.target_critic.load_state_dict(critic_params['target_critic'])
            instance.critic_optimizer.load_state_dict(critic_params['critic_optimizer'])
        return instance