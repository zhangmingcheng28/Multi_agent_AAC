import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain


class AttentionCritic(nn.Module):
    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """
    def __init__(self, sa_sizes, critic_input_dim=None, hidden_dim=32, norm_in=False, attend_heads=1):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by)
        """
        super(AttentionCritic, self).__init__()
        assert (hidden_dim % attend_heads) == 0
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)
        self.attend_heads = attend_heads

        self.critic_encoders = nn.ModuleList()
        self.critics = nn.ModuleList()

        self.state_encoders = nn.ModuleList()
        # iterate over agents
        for sdim, adim in sa_sizes:
            idim = sdim + adim
            odim = adim
            encoder = nn.Sequential()
            if norm_in:
                encoder.add_module('enc_bn', nn.BatchNorm1d(idim,
                                                            affine=False))
            encoder.add_module('enc_fc1', nn.Linear(idim, hidden_dim))
            # encoder.add_module('enc_nl', nn.LeakyReLU())
            encoder.add_module('enc_nl', nn.ReLU())
            self.critic_encoders.append(encoder)
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(2 * hidden_dim,
                                                      hidden_dim))
            # critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_nl', nn.ReLU())
            critic.add_module('critic_fc2', nn.Linear(hidden_dim, odim))
            self.critics.append(critic)

            state_encoder = nn.Sequential()
            if norm_in:
                state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(
                                            sdim, affine=False))
            state_encoder.add_module('s_enc_fc1', nn.Linear(sdim,
                                                            hidden_dim))
            # state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
            state_encoder.add_module('s_enc_nl', nn.ReLU())
            self.state_encoders.append(state_encoder)
        # ------------------------ attention for individual agents --------------------
        # # 5, is the number of agents
        # self.sum_own_fc = nn.Sequential(nn.Linear(critic_input_dim[0]*5, 1024), nn.ReLU())  # may be here can be replaced with another attention mechanism
        # self.sum_grid_fc = nn.Sequential(nn.Linear(critic_input_dim[1]*5, 256), nn.ReLU())
        #
        # self.single_own_fc = nn.Sequential(nn.Linear(critic_input_dim[0], 128), nn.ReLU())  # may be here can be replaced with another attention mechanism
        # self.single_grid_fc = nn.Sequential(nn.Linear(critic_input_dim[1], 128), nn.ReLU())
        # self.single_surr = nn.Sequential(nn.Linear(critic_input_dim[2], 128), nn.ReLU())
        # self.single_env_combine = nn.Sequential(nn.Linear(128+128, 64), nn.ReLU())
        #
        # self.single_k = nn.Linear(128, 128, bias=False)
        # self.single_q = nn.Linear(128, 128, bias=False)
        # self.single_v = nn.Linear(128, 128, bias=False)
        #
        # self.combine_env_fc = nn.Sequential(nn.Linear((5*128)+(5*128), 512), nn.ReLU())
        #
        # self.combine_all = nn.Sequential(nn.Linear(512+5 * 5, 512), nn.ReLU(),
        #                                  nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))

        # # iterate over agents
        # for _ in range(5):  # 5 is the total number of agents
        #     encoder = nn.Sequential()
        #     encoder.add_module('enc_fc1', nn.Linear(64+2, hidden_dim))
        #     encoder.add_module('enc_nl', nn.ReLU())
        #     self.critic_encoders.append(encoder)
        #
        #     critic = nn.Sequential()
        #     critic.add_module('critic_fc1', nn.Linear(2 * hidden_dim, hidden_dim))  # this 2, is hard coded from the original code
        #     critic.add_module('critic_nl', nn.ReLU())
        #     critic.add_module('critic_fc2', nn.Linear(hidden_dim, 2))  # 2 stand for the output action dimension
        #     self.critics.append(critic)
        #
        #     state_encoder = nn.Sequential()
        #     state_encoder.add_module('s_enc_fc1', nn.Linear(64, hidden_dim))
        #     state_encoder.add_module('s_enc_nl', nn.ReLU())
        #     self.state_encoders.append(state_encoder)
        # ------------------------ end of attention for individual agents --------------------

        attend_dim = hidden_dim // attend_heads
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        self.value_extractors = nn.ModuleList()
        for i in range(attend_heads):
            self.key_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.selector_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.value_extractors.append(nn.Sequential(nn.Linear(hidden_dim, attend_dim), nn.ReLU()))

        self.shared_modules = [self.key_extractors, self.selector_extractors,
                               self.value_extractors, self.critic_encoders]

    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.nagents)

    def forward(self, inps, agents=None, return_q=True, return_all_q=False,
                regularize=False, return_attend=False, niter=0):
        """
        Inputs:
            inps (list of PyTorch Matrices): Inputs to each agents' encoder
                                             (batch of obs + ac)
            agents (int): indices of agents to return Q for
            return_q (bool): return Q-value
            return_all_q (bool): return Q-value for all actions
            regularize (bool): returns values to add to loss function for
                               regularization
            return_attend (bool): return attention weights per agent
            logger (TensorboardX SummaryWriter): If passed in, important values
                                                 are logged
        """
        if agents is None:
            agents = range(len(self.critic_encoders))

        # ---------------------------------------- start of previous #
        # obtain the state after attention
        # pre-process, compute attention for every agent based on their surrounding agents
        # encoded_all_agent_state = []
        # for one_agent_batch_own, one_agent_batch_grid, one_agent_batch_surr in zip(*inps[0]):  # automatically loop over 5 times
        #     single_grid_out = self.single_grid_fc(one_agent_batch_grid)
        #     single_own_out = self.single_own_fc(one_agent_batch_own)
        #     single_surr_out = self.single_surr(one_agent_batch_surr)
        #     single_q = self.single_q(single_own_out)
        #     single_k = self.single_k(single_surr_out)
        #     single_v = self.single_v(single_surr_out)
        #     mask = one_agent_batch_surr.mean(axis=2, keepdim=True).bool()
        #     score = torch.bmm(single_k, single_q.unsqueeze(axis=2))
        #     score_mask = score.clone()  # clone操作很必要
        #     score_mask[~mask] = float('-inf')  # 不然赋值操作后会无法计算梯度
        #     alpha = F.softmax(score_mask / np.sqrt(single_k.size(-1)), dim=1)  # we use dim=1 here because we need to get attention of each sequence in K towards all hidden vector of q in each batch.
        #     alpha_mask = alpha.clone()
        #     alpha_mask[~mask] = 0
        #     v_att = torch.sum(single_v * alpha_mask, axis=1)
        #
        #     encoded_all_agent_state.append(self.single_env_combine(torch.cat((v_att, single_grid_out), dim=1)))

        # states = encoded_all_agent_state
        # actions = [a for a in inps[1]]
        # inps = [torch.cat((s, a), dim=1) for s, a in zip(encoded_all_agent_state, inps[1])]
        # # extract state-action encoding for each agent
        # sa_encodings = []
        # for encoder, inp in zip(self.critic_encoders, inps):
        #     one_sa = encoder(inp)
        #     sa_encodings.append(one_sa)
        #
        # # extract state encoding for each agent that we're returning Q for
        # s_encodings = []
        # for a_i in agents:
        #     one_s = self.state_encoders[a_i](states[a_i])
        #     s_encodings.append(one_s)
        # # s_encodings = [self.state_encoders[a_i](states[a_i]) for a_i in agents]
        # # extract keys for each head for each agent
        # all_head_keys = []
        # for k_ext in self.key_extractors:
        #     each_head_all_agent_key = []
        #     for enc in sa_encodings:
        #         one_agent_keys = k_ext(enc)
        #         each_head_all_agent_key.append(one_agent_keys)
        #     all_head_keys.append(each_head_all_agent_key)
        #
        # # all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]  # this is the original version. All values the same, thanks to same torch seed.
        #
        # # extract sa values for each head for each agent
        # all_head_values = []
        # for v_ext in self.value_extractors:
        #     each_head_all_agent_value = []
        #     for enc in sa_encodings:
        #         one_agent_value = v_ext(enc)
        #         each_head_all_agent_value.append(one_agent_value)
        #     all_head_values.append(each_head_all_agent_value)
        # # all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]  # original
        #
        # # extract selectors for each head for each agent that we're returning Q for
        # all_head_selectors = []
        # for sel_ext in self.selector_extractors:
        #     each_head_selector = []
        #     for i, enc in enumerate(s_encodings):
        #         if i in agents:
        #             one_agent_selector = sel_ext(enc)
        #             each_head_selector.append(one_agent_selector)
        #     all_head_selectors.append(each_head_selector)
        #
        # # all_head_selectors = [[sel_ext(enc) for i, enc in enumerate(s_encodings) if i in agents]for sel_ext in self.selector_extractors]  # original
        #
        # other_all_values = [[] for _ in range(len(agents))]
        # all_attend_logits = [[] for _ in range(len(agents))]
        # all_attend_probs = [[] for _ in range(len(agents))]
        # # calculate attention per head
        # for curr_head_keys, curr_head_values, curr_head_selectors in zip(
        #         all_head_keys, all_head_values, all_head_selectors):
        #     # iterate over agents
        #     for i, a_i, selector in zip(range(len(agents)), agents, curr_head_selectors):
        #         keys = [k for j, k in enumerate(curr_head_keys) if j != a_i]
        #         values = [v for j, v in enumerate(curr_head_values) if j != a_i]
        #         # calculate attention across agents
        #         attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),
        #                                      torch.stack(keys).permute(1, 2, 0))
        #         # scale dot-products by size of key (from Attention is All You Need)
        #         scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
        #         attend_weights = F.softmax(scaled_attend_logits, dim=2)
        #         other_values = (torch.stack(values).permute(1, 2, 0) *
        #                         attend_weights).sum(dim=2)
        #         other_all_values[i].append(other_values)
        #         all_attend_logits[i].append(attend_logits)
        #         all_attend_probs[i].append(attend_weights)
        # # calculate Q per agent
        # all_rets = []
        # for i, a_i in enumerate(agents):
        #     head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(1)
        #                        .mean()) for probs in all_attend_probs[i]]
        #     agent_rets = []
        #     critic_in = torch.cat((s_encodings[i], *other_all_values[i]), dim=1)
        #     all_q = self.critics[a_i](critic_in)
        #     int_acs = actions[a_i].max(dim=1, keepdim=True)[1]
        #     q = all_q.gather(1, int_acs)
        #     if return_q:
        #         agent_rets.append(q)
        #     if return_all_q:
        #         agent_rets.append(all_q)
        #     if regularize:
        #         # regularize magnitude of attention logits
        #         attend_mag_reg = 1e-3 * sum((logit**2).mean() for logit in
        #                                     all_attend_logits[i])
        #         regs = (attend_mag_reg,)
        #         agent_rets.append(regs)
        #     if return_attend:
        #         agent_rets.append(np.array(all_attend_probs[i]))
        #     # if logger is not None:
        #     #     logger.add_scalars('agent%i/attention' % a_i,
        #     #                        dict(('head%i_entropy' % h_i, ent) for h_i, ent
        #     #                             in enumerate(head_entropies)),
        #     #                        niter)
        #     if len(agent_rets) == 1:
        #         all_rets.append(agent_rets[0])  # for critic_target without regularization
        #     else:
        #         all_rets.append(agent_rets)  # for critic_prediction net, with regularization
        # if len(all_rets) == 1:
        #     return all_rets[0]
        # else:
        #     return all_rets
        # ------------------------- end of previous #
        states = [s for s, a in inps]
        actions = [a for s, a in inps]
        inps = [torch.cat((s, a), dim=1) for s, a in inps]
        # extract state-action encoding for each agent
        sa_encodings = [encoder(inp) for encoder, inp in zip(self.critic_encoders, inps)]
        # extract state encoding for each agent that we're returning Q for
        s_encodings = [self.state_encoders[a_i](states[a_i]) for a_i in agents]
        # extract keys for each head for each agent
        all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]
        # extract sa values for each head for each agent
        all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]
        # extract selectors for each head for each agent that we're returning Q for
        all_head_selectors = [[sel_ext(enc) for i, enc in enumerate(s_encodings) if i in agents]
                              for sel_ext in self.selector_extractors]

        other_all_values = [[] for _ in range(len(agents))]
        all_attend_logits = [[] for _ in range(len(agents))]
        all_attend_probs = [[] for _ in range(len(agents))]
        # calculate attention per head
        for curr_head_keys, curr_head_values, curr_head_selectors in zip(
                all_head_keys, all_head_values, all_head_selectors):
            # iterate over agents
            for i, a_i, selector in zip(range(len(agents)), agents, curr_head_selectors):
                keys = [k for j, k in enumerate(curr_head_keys) if j != a_i]
                values = [v for j, v in enumerate(curr_head_values) if j != a_i]
                # calculate attention across agents
                attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),
                                             torch.stack(keys).permute(1, 2, 0))
                # scale dot-products by size of key (from Attention is All You Need)
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
                attend_weights = F.softmax(scaled_attend_logits, dim=2)
                other_values = (torch.stack(values).permute(1, 2, 0) *
                                attend_weights).sum(dim=2)
                other_all_values[i].append(other_values)
                all_attend_logits[i].append(attend_logits)
                all_attend_probs[i].append(attend_weights)
        # calculate Q per agent
        all_rets = []
        for i, a_i in enumerate(agents):
            head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(1)
                               .mean()) for probs in all_attend_probs[i]]
            agent_rets = []
            critic_in = torch.cat((s_encodings[i], *other_all_values[i]), dim=1)
            all_q = self.critics[a_i](critic_in)  # this is the last layer of MLP
            int_acs = actions[a_i].max(dim=1, keepdim=True)[1]
            q = all_q.gather(1, int_acs)
            if return_q:
                agent_rets.append(q)
            if return_all_q:
                agent_rets.append(all_q)
            if regularize:
                # regularize magnitude of attention logits
                attend_mag_reg = 1e-3 * sum((logit**2).mean() for logit in
                                            all_attend_logits[i])
                regs = (attend_mag_reg,)
                agent_rets.append(regs)
            if return_attend:
                agent_rets.append(np.array(all_attend_probs[i]))
            # if logger is not None:
            #     logger.add_scalars('agent%i/attention' % a_i,
            #                        dict(('head%i_entropy' % h_i, ent) for h_i, ent
            #                             in enumerate(head_entropies)),
            #                        niter)
            if len(agent_rets) == 1:
                all_rets.append(agent_rets[0])
            else:
                all_rets.append(agent_rets)
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets

