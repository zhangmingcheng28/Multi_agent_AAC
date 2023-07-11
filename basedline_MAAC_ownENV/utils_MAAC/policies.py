import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from utils_MAAC.misc import onehot_from_logits, categorical_sample

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class BasePolicy(nn.Module):
    """
    Base policy network
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.leaky_relu,
                 norm_in=True, onehot_dim=0):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(BasePolicy, self).__init__()

        self.own_fc = nn.Sequential(nn.Linear(input_dim[0], 128), nn.ReLU())
        self.intrude_fc = nn.Sequential(nn.Linear(input_dim[2], 128), nn.ReLU())
        self.own_grid_fc = nn.Sequential(nn.Linear(input_dim[1], 128), nn.ReLU())
        # self.action_out_V5_1 = nn.Sequential(nn.Linear(128+128+128, 128), nn.ReLU(), nn.Linear(128, out_dim), nn.Tanh())
        # self.action_out_SAC = nn.Sequential(nn.Linear(128+128+128, 128), nn.Linear(128, out_dim))
        self.action_out_SAC = nn.Sequential(nn.Linear(128+128+128, 128), nn.Linear(128, out_dim))

        # attention for NN
        self.k = nn.Linear(128, 128, bias=False)
        self.q = nn.Linear(128, 128, bias=False)
        self.v = nn.Linear(128, 128, bias=False)

        # for SAC
        self.log_std_min = -20
        self.log_std_max = 2

        self.mean_linear = nn.Linear(128, out_dim)
        # self.mean_linear.weight.data.uniform_(-3e-3, 3e-3)
        # self.mean_linear.bias.data.uniform_(-3e-3, 3e-3)

        self.log_std_linear = nn.Linear(128, out_dim)
        # self.log_std_linear.weight.data.uniform_(-3e-3, 3e-3)
        # self.log_std_linear.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        own_e = self.own_fc(state[0])
        x_e = self.intrude_fc(state[2])

        # mask attention embedding
        q = self.q(own_e)
        k = self.k(x_e)
        v = self.v(x_e)
        mask = state[2].mean(axis=2, keepdim=True).bool()
        score = torch.bmm(k, q.unsqueeze(axis=2))
        score_mask = score.clone()  # clone操作很必要
        score_mask[~mask] = float('-inf')  # 不然赋值操作后会无法计算梯度

        alpha = F.softmax(score_mask / np.sqrt(k.size(-1)), dim=1)  # we use dim=1 here because we need to get attention of each sequence in K towards all hidden vector of q in each batch.
        alpha_mask = alpha.clone()
        alpha_mask[~mask] = 0
        v_att = torch.sum(v * alpha_mask, axis=1)

        own_grid = self.own_grid_fc(state[1])

        concat = torch.cat((own_e, v_att, own_grid), dim=1)

        # action_out = self.action_out_V5_1(concat)
        action_out = self.action_out_SAC(concat)

        # mean = self.mean_linear(action_out)
        # log_std = self.log_std_linear(action_out)

        mean = self.action_out_SAC(concat)
        log_std = self.action_out_SAC(concat)

        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        if torch.isnan(mean).any() or torch.isinf(mean).any() or torch.isnan(log_std).any() or torch.isinf(log_std).any():
            print("debug to check")
        return mean, log_std


class DiscretePolicy(BasePolicy):
    """
    Policy Network for discrete action spaces
    """
    def __init__(self, *args, **kwargs):
        super(DiscretePolicy, self).__init__(*args, **kwargs)

    def forward(self, obs, return_all_probs=False, return_log_pi=False, regularize=False):
        mean, log_std = super(DiscretePolicy, self).forward(obs)
        std = log_std.exp()
        normal = Normal(0, 1)
        z = normal.sample(mean.shape).to(device)
        action_out = torch.tanh(mean + std*z)
        rets = [action_out]

        # for calculate evaluate the pdf at each choice of action_out
        normal = torch.distributions.Normal(mean, std)
        log_probability = normal.log_prob(action_out)
        probs = log_probability.exp()  # generate pdf for each choice of action
        if return_all_probs:
            rets.append(probs)
        if return_log_pi:
            epsilon = 1e-6
            # return log probability of selected action based on SAC with re-parameterization trick
            log_prob = Normal(mean, std).log_prob(action_out) - torch.log(1 - action_out.pow(2) + epsilon)  # get the tanh adjusted log probabilities of the choosen action
            log_prob = log_prob.sum(dim=-1, keepdim=True)  # my action_out is multi-dimensional, just sum them to get the total log probability of the choose action
            rets.append(log_prob)
        if regularize:
            rets.append([(action_out**2).mean()])
        if len(rets) == 1:
            return rets[0]
        return rets
