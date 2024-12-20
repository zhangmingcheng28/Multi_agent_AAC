import sys
# sys.path.append('F:\githubClone\Multi_agent_AAC\old_framework_test')
# sys.path.append('D:\Multi_agent_AAC\old_framework_test')
from env.make_env import make_env
import argparse, datetime
import numpy as np
import torch, os
import wandb
import pandas as pd
from parameters_MADDPGv2_flowV1 import initialize_parameters
from maddpg_agent_MADDPGv2_flowV1 import MADDPG
from utils_MADDPGv2_flowV1 import *
from copy import deepcopy
import torch
import matplotlib.pyplot as plt
import matplotlib
import time
from shapely.geometry import LineString, Point, Polygon
from shapely.strtree import STRtree
from matplotlib.markers import MarkerStyle
import math
from matplotlib.transforms import Affine2D
from Utilities_own_MADDPGv2_flowV1 import *
import csv


def main(args):
    num_devices = torch.cuda.device_count()
    print("Number of GPUs:", num_devices)
    # Get the names of the available GPUs
    gpu_names = [torch.cuda.get_device_name(i) for i in range(num_devices)]
    print("GPU Names:", gpu_names)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using GPU')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    today = datetime.date.today()
    current_date = today.strftime("%d%m%y")
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%H_%M_%S")

    if args.mode == 'train':
        file_name = 'D:\MADDPG_2nd_jp/' + str(current_date) + '_' + str(formatted_time)
        if not os.path.exists(file_name):
            os.makedirs(file_name)
        plot_file_name = file_name + '/toplot'
        if not os.path.exists(plot_file_name):
            os.makedirs(plot_file_name)

    # wandb.login(key="efb76db851374f93228250eda60639c70a93d1ec")
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="MADDPG_sample_newFrameWork",
    #     name='MADDPG_test_'+str(current_date) + '_' + str(formatted_time),
    #     # track hyperparameters and run metadata
    #     config={
    #         "learning_rate": args.a_lr,
    #         "epochs": args.max_episodes,
    #     }
    # )

    # env = make_env(args.scenario)  # original environment

    # -------------- create my own environment -----------------
    n_episodes, max_t, eps_start, eps_end, eps_period, eps, env, \
    agent_grid_obs, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, learning_rate, UPDATE_EVERY, seed_used, max_xy = initialize_parameters()
    # total_agentNum = len(pd.read_excel(env.agentConfig))
    total_agentNum = 2
    # create world
    actor_dim = [6+(total_agentNum-1)*2, 9, 6]  # dim host, maximum dim grid, dim other drones
    critic_dim = [6+(total_agentNum-1)*2, 9, 6]
    n_actions = 2

    # original
    # actorNet_lr = learning_rate
    # criticNet_lr = learning_rate

    actorNet_lr = 0.0001
    # criticNet_lr = 0.0001
    criticNet_lr = 0.001

    # noise parameter ini
    largest_Nsigma = 0.5
    smallest_Nsigma = 0.15
    ini_Nsigma = largest_Nsigma

    max_spd = 15
    env.create_world(total_agentNum, n_actions, GAMMA, TAU, UPDATE_EVERY, largest_Nsigma, smallest_Nsigma, ini_Nsigma, max_xy, max_spd)

    # --------- original -----------
    # n_agents = env.n
    # n_actions = env.world.dim_p
    # n_states = env.observation_space[0].shape[0]

    # --------- my own -----------
    n_agents = len(env.all_agents)
    n_actions = n_actions


    torch.manual_seed(args.seed)  # this is the seed

    # if args.tensorboard and args.mode == "train":
    #     writer = SummaryWriter(log_dir='runs/' + args.algo + "/" + args.log_dir)

    if args.algo == "maddpg":
        model = MADDPG(actor_dim, critic_dim, n_actions, n_agents, args, criticNet_lr, actorNet_lr, GAMMA, TAU)

    # print(model)
    #model.load_model()

    episode = 0
    total_step = 0
    score_history = []
    reward_each_agent = []
    if args.mode == 'eval':
        matplotlib.use('TkAgg')
        plt.ion()
        # Create figure and axis objects
        fig, ax = plt.subplots()
        ax.set_xlim(env.bound[0], env.bound[1])  # Set x-axis limits from 0 to 5
        ax.set_ylim(env.bound[2], env.bound[3])  # Set y-axis limits from 0 to 5
    max_agent_to_add = 100-n_agents

    agent_added = 0  # this is an initialization
    # while episode < args.max_episodes:
    while (agent_added < max_agent_to_add) or (episode < args.max_episodes):
        # state = env.reset()  # original env reset
        agent_added = 0  # make sure at start of very episode the number of added agent is set to 0. Or else, the number of agent to add will accumulated across every episode.
        # ------------ my own env.reset() ------------ #
        cur_state, norm_cur_state = env.reset_world(show=0)

        # # draw occupied_poly
        # for one_poly in env.world_map_2D_polyList[0][0]:
        #     one_poly_mat = shapelypoly_to_matpoly(one_poly, True, 'y', 'b')
        #     ax.add_patch(one_poly_mat)
        # # display initial condition
        # for agentIdx, agent in env.all_agents.items():
        #     ax.text(agent.ini_pos[0], agent.ini_pos[1], agent.agent_name)
        #     # plot self_circle of the drone
        #     self_circle = Point(agent.ini_pos[0],
        #                         agent.ini_pos[1]).buffer(agent.protectiveBound, cap_style='round')
        #     grid_mat_Scir = shapelypoly_to_matpoly(self_circle, inFill=False, Edgecolor='k')
        #     ax.add_patch(grid_mat_Scir)
        #
        #     # plot drone's detection range
        #     detec_circle = Point(agent.ini_pos[0],
        #                          agent.ini_pos[1]).buffer(agent.detectionRange / 2, cap_style='round')
        #     detec_circle_mat = shapelypoly_to_matpoly(detec_circle, inFill=False, Edgecolor='g')
        #     ax.add_patch(detec_circle_mat)
        #
        #     # link individual drone's starting position with its goal
        #     ini = agent.ini_pos
        #     for wp in agent.goal:
        #         # plt.plot(wp[0], wp[1], marker='*', color='y', markersize=10)
        #         plt.plot([wp[0], ini[0]], [wp[1], ini[1]], '--', color='c')
        #         ini = wp

        # # Ensure the plot window is displayed
        # plt.show(block=False)

        episode += 1
        # print("current episode is {}, scaling factor is {}".format(episode, model.var[0]))
        step = 0
        accum_reward = 0
        trajectory_eachPlay = []

        # load_filepath_0 = r'D:\MADDPG_2nd_jp\190623_21_58_17\interval_record_eps\episode_11000_agent_0actor_net.pth'
        # load_filepath_1 = r'D:\MADDPG_2nd_jp\190623_21_58_17\interval_record_eps\episode_11000_agent_1actor_net.pth'
        # if args.mode == "eval":
        #     model.load_model(load_filepath_0, load_filepath_1)
        while True:
            if args.mode == "train":
                cur_state, norm_cur_state = env.fill_agents(n_agents, cur_state, norm_cur_state)  # if a new agent is filled, we need to update the state information
                action = model.choose_action(norm_cur_state, episode, env, noisy=True)
                next_state, norm_next_state = env.step(action, step)
                reward_aft_action, done_aft_action, check_goal, agent_to_remove = env.get_step_reward(step)  # remove agent here
                new_length = len(agent_to_remove)  # check if length of agnet_to_remove increased during each step
                agent_added = agent_added + new_length
                # reward_aft_action, done_aft_action, check_goal = env.get_step_reward_5_v3(step)

                step += 1  # current play step
                total_step += 1  # steps taken from 1st episode

                if args.algo == "maddpg" or args.algo == "commnet":
                    # obs = torch.from_numpy(np.stack(cur_state)).float().to(device)
                    # obs = torch.from_numpy(np.stack(norm_cur_state)).float().to(device)
                    obs = [torch.from_numpy(np.stack(element)).data.float().to(device) for element in norm_cur_state]
                    # obs_ = torch.from_numpy(np.stack(next_state)).float().to(device)
                    # obs_ = torch.from_numpy(np.stack(norm_next_state)).float().to(device)
                    next_obs = [torch.from_numpy(np.stack(element)).data.float().to(device) for element in norm_next_state]

                    rw_tensor = torch.FloatTensor(np.array(reward_aft_action)).to(device)
                    # convert "action" from dict to array, so that can be stored in a experience replay
                    action = np.array(list(action.values()))
                    ac_tensor = torch.FloatTensor(action).to(device)
                    done_tensor = torch.FloatTensor(done_aft_action).to(device)
                    if args.algo == "commnet" and next_obs is not None:
                        model.memory.push(obs.data, ac_tensor, next_obs, rw_tensor, done_tensor)
                    if args.algo == "maddpg":
                        model.memory.push(obs, ac_tensor, next_obs, rw_tensor, done_tensor)
                else:
                    model.memory(cur_state, action, reward_aft_action, next_state, done_aft_action)

                accum_reward = accum_reward + reward_aft_action[0]

                c_loss, a_loss = model.update_myown(episode, total_step,
                                                    UPDATE_EVERY)  # last working learning framework

                cur_state = next_state
                norm_cur_state = norm_next_state

                if args.episode_length < step or (True in done_aft_action):
                    # display bound lines
                    # display condition of failing
                    # here onwards is end of an episode's play
                    score_history.append(accum_reward)
                    print("[Episode %05d] reward %6.4f" % (episode, accum_reward))
                    # wandb.log({'overall_reward': float(accum_reward)})
                    if c_loss and a_loss:
                        for idx, val in enumerate(c_loss):
                            print(" agent %s, a_loss %3.2f c_loss %3.2f" % (
                                idx, a_loss[idx].item(), c_loss[idx].item()))
                            # wandb.log({'agent' + str(idx) + 'actor_loss': float(a_loss[idx].item())})
                            # wandb.log({'agent' + str(idx) + 'critic_loss': float(c_loss[idx].item())})
                    if episode % args.save_interval == 0 and args.mode == "train":

                        # save the models at a predefined interval
                        # save model to my own directory
                        filepath = file_name+'/interval_record_eps'
                        model.save_model(episode, filepath)  # this is the original save model

                    # cur_state, norm_cur_state = env.reset_world(show=0)
                    # model.reset()
                    break  # this is to break out from "while True:", which is one play
            elif args.mode == "eval":

                ax.set_xlim(env.bound[0], env.bound[1])  # Set x-axis limits from 0 to 5
                ax.set_ylim(env.bound[2], env.bound[3])  # Set y-axis limits from 0 to 5

                # action = model.choose_action(norm_cur_state, episode, noisy=False)
                cur_state, norm_cur_state = env.fill_agents(n_agents, cur_state, norm_cur_state)
                action = env.get_actions_noCR()  # only update heading
                next_state, norm_next_state = env.step(action, step)
                reward_aft_action, done_aft_action, check_goal, agent_to_remove = env.get_step_reward(step)
                new_length = len(agent_to_remove)  # check if length of agnet_to_remove increased during each step
                agent_added = agent_added + new_length
                # reward_aft_action, done_aft_action, check_goal = env.get_step_reward_5_v3(step)

                step += 1
                total_step += 1
                cur_state = next_state
                norm_cur_state = norm_next_state
                trajectory_eachPlay.append([[each_agent_traj[0], each_agent_traj[1]] for each_agent_traj in cur_state[0]])
                accum_reward = accum_reward + sum(reward_aft_action)

                # draw occupied_poly
                for one_poly in env.world_map_2D_polyList[0][0]:
                    one_poly_mat = shapelypoly_to_matpoly(one_poly, True, 'y', 'b')
                    ax.add_patch(one_poly_mat)
                # display  condition
                for agentIdx, agent in env.all_agents.items():
                    ax.text(agent.pos[0], agent.pos[1], agent.agent_name)
                    # plot self_circle of the drone
                    self_circle = Point(agent.pos[0],
                                        agent.pos[1]).buffer(agent.protectiveBound, cap_style='round')
                    grid_mat_Scir = shapelypoly_to_matpoly(self_circle, inFill=False, Edgecolor='k')
                    ax.add_patch(grid_mat_Scir)
                    # plot drone's detection range
                    detec_circle = Point(agent.pos[0],
                                         agent.pos[1]).buffer(agent.detectionRange / 2, cap_style='round')
                    detec_circle_mat = shapelypoly_to_matpoly(detec_circle, inFill=False, Edgecolor='g')
                    ax.add_patch(detec_circle_mat)

                print("current added agent is {}".format(agent_added))
                fig.canvas.draw()
                fig.canvas.flush_events()
                # Pause for 0.01 seconds
                time.sleep(0.01)
                ax.cla()

                if args.episode_length < step or (True in done_aft_action) or (agent_added > max_agent_to_add):  # when termination condition reached
                    print("[Episode %05d] reward %6.4f " % (episode, accum_reward))
                    # display trajectory
                    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
                    matplotlib.use('TkAgg')
                    fig, ax = plt.subplots(1, 1)
                    # display initial condition
                    for agentIdx, agent in env.all_agents.items():

                        plt.plot(agent.ini_pos[0], agent.ini_pos[1],
                                 marker=MarkerStyle(">",
                                                    fillstyle="right",
                                                    transform=Affine2D().rotate_deg(math.degrees(agent.heading))),
                                 color='g')
                        plt.text(agent.ini_pos[0], agent.ini_pos[1], agent.agent_name)
                        # plot self_circle of the drone
                        self_circle = Point(agent.ini_pos[0],
                                            agent.ini_pos[1]).buffer(agent.protectiveBound, cap_style='round')
                        grid_mat_Scir = shapelypoly_to_matpoly(self_circle, inFill=False, Edgecolor='k')
                        ax.add_patch(grid_mat_Scir)

                        # plot drone's detection range
                        detec_circle = Point(agent.ini_pos[0],
                                             agent.ini_pos[1]).buffer(agent.detectionRange / 2, cap_style='round')
                        detec_circle_mat = shapelypoly_to_matpoly(detec_circle, inFill=False, Edgecolor='g')
                        ax.add_patch(detec_circle_mat)

                        # link individual drone's starting position with its goal
                        ini = agent.ini_pos
                        for wp in agent.goal:
                            plt.plot(wp[0], wp[1], marker='*', color='y', markersize=10)
                            plt.plot([wp[0], ini[0]], [wp[1], ini[1]], '--', color='c')
                            ini = wp

                    # draw trajectory in current episode
                    for trajectory_idx, trajectory_val in enumerate(trajectory_eachPlay):  # each time step
                        for agentIDX, each_agent_traj in enumerate(trajectory_val):  # for each agent's motion in a time step
                            x, y = each_agent_traj[0], each_agent_traj[1]
                            plt.plot(x, y, 'o', color='r')

                            # plt.text(x-1, y-1, str(round(float(reward_each_agent[trajectory_idx][agentIDX]),2)))

                            self_circle = Point(x, y).buffer(env.all_agents[0].protectiveBound, cap_style='round')
                            grid_mat_Scir = shapelypoly_to_matpoly(self_circle, False, 'k')
                            ax.add_patch(grid_mat_Scir)

                    # draw occupied_poly
                    for one_poly in env.world_map_2D_polyList[0][0]:
                        one_poly_mat = shapelypoly_to_matpoly(one_poly, True, 'y', 'b')
                        ax.add_patch(one_poly_mat)
                    # draw non-occupied_poly
                    for zero_poly in env.world_map_2D_polyList[0][1]:
                        zero_poly_mat = shapelypoly_to_matpoly(zero_poly, False, 'y')
                        # ax.add_patch(zero_poly_mat)

                    # show building obstacles
                    for poly in env.buildingPolygons:
                        matp_poly = shapelypoly_to_matpoly(poly, False, 'red')  # the 3rd parameter is the edge color
                        ax.add_patch(matp_poly)

                    plt.axis('equal')
                    plt.xlim(env.bound[0], env.bound[1])
                    plt.ylim(env.bound[2], env.bound[3])
                    plt.axvline(x=env.bound[0], c="green")
                    plt.axvline(x=env.bound[1], c="green")
                    plt.axhline(y=env.bound[2], c="green")
                    plt.axhline(y=env.bound[3], c="green")
                    plt.xlabel("X axis")
                    plt.ylabel("Y axis")
                    plt.show()
                    break

    # wandb.finish()

    # if args.tensorboard:
    #     writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default="simple_spread", type=str)
    parser.add_argument('--max_episodes', default=200000, type=int)  # rnu for a total of 60000 episodes
    parser.add_argument('--algo', default="maddpg", type=str, help="commnet/bicnet/maddpg")
    parser.add_argument('--mode', default="train", type=str, help="train/eval")
    parser.add_argument('--episode_length', default=800, type=int)  # maximum play per episode
    parser.add_argument('--memory_length', default=int(1e5), type=int)
    parser.add_argument('--tau', default=0.001, type=float)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--a_lr', default=0.0001, type=float)
    parser.add_argument('--c_lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=512, type=int)  # original 256
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
