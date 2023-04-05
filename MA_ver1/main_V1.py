# -*- coding: utf-8 -*-
"""
@Time    : 3/1/2023 7:57 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
from parameters_V1 import initialize_parameters
from Multi_Agent_replaybuffer_V1 import MultiAgentReplayBuffer
from shapely.geometry import LineString, Point, Polygon
from shapely.strtree import STRtree
import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.transforms import Affine2D
import pickle
from matplotlib.markers import MarkerStyle
import math
import time
import random
from Utilities_V1 import sort_polygons, shapelypoly_to_matpoly, \
    extract_individual_obs, map_range, compute_potential_conflict

if __name__ == '__main__':
    start_time = time.time()
    # initialize parameters
    n_episodes, max_t, eps_start, eps_end, eps_period, eps, env, \
    agent_grid_obs, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, learning_rate, UPDATE_EVERY, seed_used = initialize_parameters()
    train_eva = "train"
    random.seed(seed_used)
    # set number of drone in the airspace
    total_agentNum = 5
    # create world
    actor_obs = [6, 20, 6]  # dim host, maximum dim grid, dim other drones
    critic_obs = [6, 20, 6]
    n_actions = 2
    actorNet_lr = learning_rate
    criticNet_lr = learning_rate
    # create agents, reset environment
    env.create_world(total_agentNum, critic_obs, actor_obs, n_actions, actorNet_lr, criticNet_lr, GAMMA, TAU)

    # initialized memory replay
    actor_dims = 3  # A list of 3 list, each 1st list has length 3, 2nd has length 20, 3rd has length 6
    critic_dims = total_agentNum * actor_dims  # critic is centralized, so we combine dim of all agents
    ReplayBuffer = MultiAgentReplayBuffer(BUFFER_SIZE, actor_dims, critic_dims, total_agentNum, n_actions, batch_size=BATCH_SIZE)
    print("time to initiate is {}".format(time.time()-start_time))
    score_history = []
    Trajectory_history = []

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
        print("Episode {} starts".format(i))
        episode_score = 0  # Each episode, this is a sum of rewards at individual step of a play
        trajectory_eachPlay = []

        cur_state = env.reset_world(show=0)

        for ts in range(max_t):  # steps inside an episode
            #  get action, no CR is used, output is the velocity
            #  actions, noCR = env.get_actions_noCR(combine_state)
            start_action_time = time.time()
            #  get action with neural networks
            actions = env.get_actions_NN(cur_state)
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
                critic_losses, actor_losses = env.central_learning(ReplayBuffer, BATCH_SIZE)

            # record reward each step
            episode_score = episode_score + sum(reward_aft_action)
            # record trajectory for each drone in the play
            trajectory_eachPlay.append(cur_state[0])

            # propagate the environment
            cur_state = next_state
            # recording of trajectory and score must before the "break" action, so that the collision
            # or goal reaching step can be recorded
            if 1 in done_aft_action:
                # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
                # matplotlib.use('TkAgg')
                # fig, ax = plt.subplots(1, 1)
                # for agentIdx, agent in env.all_agents.items():
                #     plt.plot(agent.ini_pos[0], agent.ini_pos[1], marker=MarkerStyle(">", fillstyle="right",
                #                                                             transform=Affine2D().rotate_deg(
                #                                                                 math.degrees(agent.heading))),
                #              color='y')
                #     plt.text(agent.ini_pos[0], agent.ini_pos[1], agent.agent_name)
                #     # plot self_circle of the drone
                #     self_circle = Point(agent.ini_pos[0], agent.ini_pos[1]).buffer(agent.protectiveBound, cap_style='round')
                #     grid_mat_Scir = shapelypoly_to_matpoly(self_circle, False, 'k')
                #     ax.add_patch(grid_mat_Scir)
                #
                #     # plot drone's detection range
                #     detec_circle = Point(agent.ini_pos[0], agent.ini_pos[1]).buffer(agent.detectionRange / 2, cap_style='round')
                #     detec_circle_mat = shapelypoly_to_matpoly(detec_circle, False, 'g')
                #     ax.add_patch(detec_circle_mat)
                #
                #     ini = agent.ini_pos
                #     for wp in agent.goal:
                #         plt.plot(wp[0], wp[1], marker='*', color='y', markersize=10)
                #         plt.plot([wp[0], ini[0]], [wp[1], ini[1]], '--', color='c')
                #         ini = wp
                #
                # # draw trajectory in current episode
                # for trajectory in trajectory_eachPlay:  # each time step
                #     for each_agent_idx in range(trajectory.shape[0]):  # for each agent's motion in a time step
                #         x, y = trajectory[each_agent_idx, 0], trajectory[each_agent_idx, 1]
                #         plt.plot(x, y, 'o', color='r')
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

        # here onwards is end of an episode's play
        score_history.append(episode_score)
        Trajectory_history.append(trajectory_eachPlay)

    print('done')

