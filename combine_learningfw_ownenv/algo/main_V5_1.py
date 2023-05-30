# -*- coding: utf-8 -*-
"""
@Time    : 3/1/2023 7:57 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import csv
import wandb
from parameters_V5_1 import initialize_parameters
from Multi_Agent_replaybuffer_V5_1 import MultiAgentReplayBuffer
from shapely.geometry import LineString, Point, Polygon
from shapely.strtree import STRtree
import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.transforms import Affine2D
import pickle
import numpy as np
from matplotlib.markers import MarkerStyle
import math
import time
import random
import datetime
from Utilities_V5_1 import sort_polygons, shapelypoly_to_matpoly, \
    extract_individual_obs, map_range, compute_potential_conflict, display_trajectory, action_selection_statistics

# NOTE change batch_size and change update rate, update count go with agent class
if __name__ == '__main__':
    start_time = time.time()
    # wandb.login(key="efb76db851374f93228250eda60639c70a93d1ec")
    # initialize parameters
    n_episodes, max_t, eps_start, eps_end, eps_period, eps, env, \
    agent_grid_obs, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, learning_rate, UPDATE_EVERY, seed_used = initialize_parameters()
    train_eva = "train"
    random.seed(seed_used)
    # set number of drone in the airspace
    total_agentNum = 5
    max_nei_num = 5
    # create world
    actor_obs = [6+(total_agentNum-1)*2, 20, 6]  # dim host, maximum dim grid, dim other drones
    critic_obs = [6+(total_agentNum-1)*2, 20, 6]
    n_actions = 2
    actorNet_lr = learning_rate
    criticNet_lr = learning_rate
    # noise parameter ini
    largest_Nsigma = 0.15
    smallest_Nsigma = 0.01
    ini_Nsigma = largest_Nsigma

    # create agents, reset environment
    env.create_world(total_agentNum, critic_obs, actor_obs, n_actions, actorNet_lr, criticNet_lr, GAMMA, TAU, UPDATE_EVERY, largest_Nsigma, smallest_Nsigma, ini_Nsigma, max_nei_num)

    # initialized memory replay
    actor_dims = 3  # A list of 3 list, each 1st list has length 3, 2nd has length 20, 3rd has length 6
    critic_dims = total_agentNum * actor_dims  # critic is centralized, so we combine dim of all agents
    ReplayBuffer = MultiAgentReplayBuffer(BUFFER_SIZE, actor_dims, critic_dims, total_agentNum, n_actions, batch_size=BATCH_SIZE)
    #print("time to initiate is {}".format(time.time()-start_time))
    score_history = []
    Trajectory_history = []
    Trajectory_action_record = []

    # # get navigate to plot file and load pickle
    # with open(r'D:\MADDPG_2nd_jp\240423_16_13_47\toplot\all_episode_trajectory.pickle', 'rb') as handle:
    #     all_trajectory = pickle.load(handle)
    #
    # with open(r'D:\MADDPG_2nd_jp\270423_15_15_28\toplot\all_episode_action_taken.pickle', 'rb') as handle:
    #     action_collection = pickle.load(handle)
    #
    # # display_trajectory(env, all_trajectory)
    # action_selection_statistics(action_collection)

    # simulation result saving
    today = datetime.date.today()
    current_date = today.strftime("%d%m%y")
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%H_%M_%S")
    file_name = 'D:\MADDPG_2nd_jp/' + str(current_date) + '_' + str(formatted_time)
    if not os.path.exists(file_name):
        os.makedirs(file_name)
    plot_file_name = file_name + '/toplot'
    if not os.path.exists(plot_file_name):
        os.makedirs(plot_file_name)

    wandb.init(
        # set the wandb project where this run will be logged
        project="MADDPG_fixedDroneNum_env",
        name='MADDPG_test_'+str(current_date) + '_' + str(formatted_time),
        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "epochs": n_episodes,
        }
    )

    score_best_avg = float("-inf")
    actor_losses = None
    critic_losses = None

    # simulation / episode start

    # # critic network test
    # test_critic = env.all_agents[0].criticNet.forward(combine_state, actor_obs)

    # # check on why there is a collision
    # with open('trajectory.pickle', 'rb') as handle:
    #     trajectory_list = pickle.load(handle)
    # #     # check each wp, which wp caused termination
    #     allBuildingSTR = STRtree(env.world_map_2D_polyList[0][0])
    #     step4_agent4 = trajectory_list[-2][-1]
    #     step5_agent4 = trajectory_list[-1][-1]
    #
    #     host_pass_line = LineString([step4_agent4[0:2], step5_agent4[0:2]])
    #     host_passed_volume = host_pass_line.buffer(2.5, cap_style='round')
    #
    #     tar_circle = Point(step4_agent4[2:4]).buffer(1, cap_style='round')
    #     possiblePoly = allBuildingSTR.query(host_passed_volume)
    #     for element in possiblePoly:
    #         if allBuildingSTR.geometries.take(element).intersection(host_passed_volume):
    #             print("done crash into building cause termination")
    #             collide_building = 1
    #             break
    #
    #     os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    #     matplotlib.use('TkAgg')
    #     fig, ax = plt.subplots(1, 1)
    #     # draw trajectory in current episode
    #     drone_volume = shapelypoly_to_matpoly(host_passed_volume, True, 'r', 'm')
    #     ax.add_patch(drone_volume)
    #
    #     # draw occupied_poly
    #     for one_poly in env.world_map_2D_polyList[0][0]:
    #         one_poly_mat = shapelypoly_to_matpoly(one_poly, True, 'y', 'b')
    #         ax.add_patch(one_poly_mat)
    #     # draw non-occupied_poly
    #     for zero_poly in env.world_map_2D_polyList[0][1]:
    #         zero_poly_mat = shapelypoly_to_matpoly(zero_poly, False, 'y')
    #         # ax.add_patch(zero_poly_mat)
    #
    #     # show building obstacles
    #     for poly in env.buildingPolygons:
    #         matp_poly = shapelypoly_to_matpoly(poly, False, 'red')  # the 3rd parameter is the edge color
    #         ax.add_patch(matp_poly)
    #
    #     plt.axis('equal')
    #     plt.xlim(env.bound[0], env.bound[1])
    #     plt.ylim(env.bound[2], env.bound[3])
    #     plt.axvline(x=env.bound[0], c="green")
    #     plt.axvline(x=env.bound[1], c="green")
    #     plt.axhline(y=env.bound[2], c="green")
    #     plt.axhline(y=env.bound[3], c="green")
    #     plt.xlabel("X axis")
    #     plt.ylabel("Y axis")
    #     plt.show()

    for i in range(n_episodes):
        episode_score = 0  # Each episode, this is a sum of rewards at individual step of a play
        trajectory_eachPlay = []
        action_eachPlay = []
        reward_each_agent = []
        cur_state = env.reset_world(show=0)

        for ts in range(max_t):  # steps inside an episode
            #print("Episode {}, current time step is {}".format(i, ts))
            #  get action, no CR is used, output is the velocity
            #  actions, noCR = env.get_actions_noCR(combine_state)
            start_action_time = time.time()
            #  get action with neural networks
            actions = env.get_actions_NN(cur_state, eps)
            # proceed with the environment step, should output the new / next combine_state
            # after moving one step, every single drone should re-scan their surroundings to ensure they have capture
            # change in their surrounding neighbor changes
            next_state = env.step(actions, ts)
            # print("time used for choose action and propagate environment is {}".format(time.time()-start_action_time))
            # when every drone has taken an action we record the reward for the step taken
            reward_aft_action, done_aft_action = env.get_step_reward(ts)
            # add current play result to experience replay
            ReplayBuffer.add(cur_state, actions, reward_aft_action, next_state, done_aft_action)
            if len(ReplayBuffer.memory) >= BATCH_SIZE:
                # critic_losses, actor_losses = env.central_learning_v2(ReplayBuffer, BATCH_SIZE, total_agentNum, actor_obs[2])  # this is the new framework
                # critic_losses, actor_losses = env.central_learning(ReplayBuffer, BATCH_SIZE, total_agentNum,
                #                                                    actor_obs[2], UPDATE_EVERY)  # this is old framework
                critic_losses, actor_losses = env.central_update(ReplayBuffer, BATCH_SIZE, total_agentNum, actor_obs[2], ts)  # this is version two of the new framework
            # record reward each step
            episode_score = episode_score + sum(reward_aft_action)

            # propagate the environment
            cur_state = next_state

            # record trajectory for each drone in the play
            # trajectory_eachPlay.append([each_agent_traj[0] for each_agent_traj in cur_state])
            # the original position is not counted in the time step,when ts=0, is the 1st step drone takes.
            trajectory_eachPlay.append([[each_agent_traj[0], each_agent_traj[1]] for each_agent_traj in cur_state])
            action_eachPlay.append(actions)
            reward_each_agent.append(reward_aft_action)
            # recording of trajectory and score must before the "break" action, so that the collision
            # or goal reaching step can be recorded

            if 1 in done_aft_action:
                # # display trajectory of all agents in the environment
                # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
                # matplotlib.use('TkAgg')
                # fig, ax = plt.subplots(1, 1)
                # # display initial condition
                # for agentIdx, agent in env.all_agents.items():
                #
                #     plt.plot(agent.ini_pos[0], agent.ini_pos[1],
                #              marker=MarkerStyle(">",
                #                                 fillstyle="right",
                #                                 transform=Affine2D().rotate_deg(math.degrees(agent.heading))),
                #              color='y')
                #     plt.text(agent.ini_pos[0], agent.ini_pos[1], agent.agent_name)
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
                #         plt.plot(wp[0], wp[1], marker='*', color='y', markersize=10)
                #         plt.plot([wp[0], ini[0]], [wp[1], ini[1]], '--', color='c')
                #         ini = wp
                #
                # # draw trajectory in current episode
                # for trajectory_idx, trajectory_val in enumerate(trajectory_eachPlay):  # each time step
                #     for agentIDX, each_agent_traj in enumerate(trajectory_val):  # for each agent's motion in a time step
                #         x, y = each_agent_traj[0], each_agent_traj[1]
                #         plt.plot(x, y, 'o', color='r')
                #
                #         plt.text(x-1, y-1, str(round(float(reward_each_agent[trajectory_idx][agentIDX]),2)))
                #
                #         self_circle = Point(x, y).buffer(env.all_agents[0].protectiveBound, cap_style='round')
                #         grid_mat_Scir = shapelypoly_to_matpoly(self_circle, False, 'k')
                #         ax.add_patch(grid_mat_Scir)
                #
                # # draw occupied_poly
                # for one_poly in env.world_map_2D_polyList[0][0]:
                #     one_poly_mat = shapelypoly_to_matpoly(one_poly, True, 'y', 'b')
                #     ax.add_patch(one_poly_mat)
                # # draw non-occupied_poly
                # for zero_poly in env.world_map_2D_polyList[0][1]:
                #     zero_poly_mat = shapelypoly_to_matpoly(zero_poly, False, 'y')
                #     # ax.add_patch(zero_poly_mat)
                #
                # # show building obstacles
                # for poly in env.buildingPolygons:
                #     matp_poly = shapelypoly_to_matpoly(poly, False, 'red')  # the 3rd parameter is the edge color
                #     ax.add_patch(matp_poly)
                #
                # plt.axis('equal')
                # plt.xlim(env.bound[0], env.bound[1])
                # plt.ylim(env.bound[2], env.bound[3])
                # plt.axvline(x=env.bound[0], c="green")
                # plt.axvline(x=env.bound[1], c="green")
                # plt.axhline(y=env.bound[2], c="green")
                # plt.axhline(y=env.bound[3], c="green")
                # plt.xlabel("X axis")
                # plt.ylabel("Y axis")
                # plt.show()
                break  # terminate current episode.

        # check eps decay
        print("current episode is {}, current eps is {}, current sigma is {}".format(i, eps, env.OU_noise.sigma))
        # compute eps-decay
        eps = max(eps_end, eps - (eps_start - eps_end) / eps_period)

        # here onwards is end of an episode's play
        score_history.append(episode_score)
        Trajectory_history.append(trajectory_eachPlay)
        Trajectory_action_record.append(action_eachPlay)

        # save episodes reward for entire system
        with open(plot_file_name+'/episodes_reward.csv', 'w+') as f:
            write = csv.writer(f)
            write.writerows([score_history])
            print('episode', i, 'episode score is {:.1f}'.format(episode_score))
        wandb.log({'overall_reward': float(episode_score)})

        # save all the trajectory history
        with open(plot_file_name+'/all_episode_trajectory.pickle', 'wb') as handle:
            pickle.dump(Trajectory_history, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # save all the actions taken by all agents in 5000 episodes
        with open(plot_file_name+'/all_episode_action_taken.pickle', 'wb') as handle:
            pickle.dump(Trajectory_action_record, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # save episodes loss for entire system
        if (actor_losses != None) & (critic_losses != None):
            for individual_actor, individual_critic, agent_obj in zip(actor_losses, critic_losses, env.all_agents.values()):
                with open(plot_file_name + '/' + agent_obj.agent_name + 'actor_loss.csv', 'w') as f:
                    write = csv.writer(f)
                    write.writerows([[float(individual_actor)]])
                    # log metrics to wandb
                    wandb.log({agent_obj.agent_name + 'actor_loss': float(individual_actor)})
                with open(plot_file_name + '/' + agent_obj.agent_name + 'critic_loss.csv', 'w') as f:
                    write = csv.writer(f)
                    write.writerows([[float(individual_critic)]])
                    wandb.log({agent_obj.agent_name + 'critic_loss': float(individual_critic)})

        # get average score for the past 100 episode
        score_avg = np.mean(score_history[-100:])
        if score_avg > score_best_avg:
            # save best average score check point
            score_best_avg = score_avg
            # save actor's model when best average appears model
            env.save_model_actor_net(file_name)

        # save the models at a predefined interval
        if (i+1) % 100 == 0:
            env.save_model_actor_net(file_name+'/interval_record_eps_'+str(i))
            print('episode', i, 'average score {:.1f}'.format(score_avg))
    print('done')
    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()

