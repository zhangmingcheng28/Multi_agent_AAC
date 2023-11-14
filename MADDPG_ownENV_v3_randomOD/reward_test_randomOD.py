# -*- coding: utf-8 -*-
"""
@Time    : 7/27/2023 9:44 AM
@Author  : Thu Ra
@FileName: 
@Description: 
@Package dependency:
"""
import numpy as np
import pandas as pd
import torch
from parameters_MADDPGv3_randomOD import initialize_parameters
from maddpg_agent_MADDPGv3_randomOD import MADDPG
from matplotlib.markers import MarkerStyle
import math
from matplotlib.transforms import Affine2D
from Utilities_own_MADDPGv3_randomOD import *
import argparse, datetime
from parameters_MADDPGv3_randomOD import initialize_parameters


def main(args):
    # -------------- create my own environment -----------------
    n_episodes, max_t, eps_start, eps_end, eps_period, eps, env, \
    agent_grid_obs, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, learning_rate, UPDATE_EVERY, seed_used, max_xy = initialize_parameters()
    total_agentNum = len(pd.read_excel(env.agentConfig))
    max_nei_num = 5

    actor_dim = [6, 9, 6]  # dim host, maximum dim grid, dim other drones
    critic_dim = [6, 9, 6]
    n_actions = 2


    actorNet_lr = 0.0001
    criticNet_lr = 0.001

    # noise parameter ini
    largest_Nsigma = 0.5
    smallest_Nsigma = 0.15
    ini_Nsigma = largest_Nsigma

    max_spd = 15
    env.create_world(total_agentNum, n_actions, GAMMA, TAU, UPDATE_EVERY, largest_Nsigma, smallest_Nsigma, ini_Nsigma,
                     max_xy, max_spd)
    #
    # --------- my own -----------
    n_agents = len(env.all_agents)
    n_actions = n_actions

    torch.manual_seed(args.seed)  # this is the seed

    # if args.tensorboard and args.mode == "train":
    #     writer = SummaryWriter(log_dir='runs/' + args.algo + "/" + args.log_dir)

    if args.algo == "maddpg":
        model = MADDPG(actor_dim, critic_dim, n_actions, n_agents, args, criticNet_lr, actorNet_lr, GAMMA, TAU)

    episode = 0
    total_step = 0
    score_history = []
    eps_reward_record = []
    trajectory_eachPlay = []
    step = 0
    accum_reward = 0
    while True:
        cur_state, norm_cur_state = env.reset_world(show=0)
        step_reward_record = [None] * n_agents
        env.all_agents[0].pos[0] = 535
        env.all_agents[0].pos[1] = 355
        env.all_agents[0].pre_pos = np.array([535, 355])
        reward_aft_action, done_aft_action, check_goal, step_reward_record = env.get_step_reward_5_v3(step, step_reward_record)

        step += 1
        total_step += 1

        trajectory_eachPlay.append([[each_agent_traj[0], each_agent_traj[1]] for each_agent_traj in cur_state[0]])
        accum_reward = accum_reward + reward_aft_action[0]
        # if args.episode_length < step or (True in done_aft_action):  # when termination condition reached
        #     print('done')
        #     break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default="simple_spread", type=str)
    parser.add_argument('--max_episodes', default=15000, type=int)  # rnu for a total of 60000 episodes
    parser.add_argument('--algo', default="maddpg", type=str, help="commnet/bicnet/maddpg")
    parser.add_argument('--mode', default="train", type=str, help="train/eval")
    parser.add_argument('--episode_length', default=50, type=int)  # maximum play per episode
    parser.add_argument('--memory_length', default=int(1e5), type=int)
    parser.add_argument('--tau', default=0.001, type=float)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--a_lr', default=0.0001, type=float)
    parser.add_argument('--c_lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=512, type=int)  # original 512
    parser.add_argument('--render_flag', default=False, type=bool)
    parser.add_argument('--ou_theta', default=0.15, type=float)
    parser.add_argument('--ou_mu', default=0.0, type=float)
    parser.add_argument('--ou_sigma', default=0.2, type=float)
    parser.add_argument('--epsilon_decay', default=10000, type=int)
    parser.add_argument('--tensorboard', default=True, action="store_true")
    parser.add_argument("--save_interval", default=1000, type=int)  # save model for every 5000 episodes
    parser.add_argument("--model_episode", default=60000, type=int)
    parser.add_argument('--episode_before_train', default=10, type=int)  # original 1000
    parser.add_argument('--log_dir', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

    args = parser.parse_args()

    main(args)