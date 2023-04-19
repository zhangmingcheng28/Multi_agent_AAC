# -*- coding: utf-8 -*-
"""
@Time    : 3/2/2023 7:42 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import copy

from shapely.strtree import STRtree
from shapely.geometry import LineString, Point, Polygon
from scipy.spatial import KDTree
from copy import deepcopy
from agent_V1 import Agent
import pandas as pd
import math
import numpy as np
import os
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
import matplotlib.pyplot as plt
import matplotlib
import re
import time
from Utilities_V1 import sort_polygons, shapelypoly_to_matpoly, \
    extract_individual_obs, map_range, compute_potential_conflict, padding_list, preprocess_batch_for_critic_net, OUNoise
import torch as T
import torch
import torch.nn.functional as F


class env_simulator:
    def __init__(self, world_map, building_polygons, grid_length, bound, allGridPoly):  # allGridPoly[0][0] is all grid=1
        self.world_map_2D = world_map
        self.world_map_2D_polyList = allGridPoly
        self.gridlength = grid_length
        self.buildingPolygons = building_polygons
        self.bound = bound
        self.global_time = 0.0  # in sec
        self.time_step = 0.5  # in second as well
        self.all_agents = None
        self.cur_allAgentCoor_KD = None
        self.OU_noise = None

    def create_world(self, total_agentNum, critic_obs, actor_obs, n_actions, actorNet_lr, criticNet_lr, gamma, tau):
        # config OU_noise
        self.OU_noise = OUNoise(n_actions)
        self.all_agents = {}
        for agent_i in range(total_agentNum):
            agent = Agent(actor_obs, critic_obs, n_actions, agent_i, total_agentNum, actorNet_lr, criticNet_lr, gamma, tau)
            self.all_agents[agent_i] = agent
        global_state = self.reset_world(show=0)

    def reset_world(self, show):  # set initialize position and observation for all agents
        self.global_time = 0.0
        self.time_step = 0.5
        # reset OU_noise as well
        self.OU_noise.reset()
        # prepare for output states
        overall_state = []

        #  custom agent position
        # x-bound: [0, 1800), y-bound: [0, 1300)
        # read the Excel file into a pandas dataframe
        df = pd.read_excel(r'F:\githubClone\Multi_agent_AAC\MA_ver1\fixedDrone.xlsx')
        # df = pd.read_excel(r'D:\Multi_agent_AAC\MA_ver1\fixedDrone.xlsx')
        # convert the dataframe to a NumPy array
        custom_agent_data = np.array(df)
        custom_agent_data = custom_agent_data.astype(float)
        agentsCoor_list = []  # for store all agents as circle polygon
        agentRefer_dict = {}  # A dictionary to use agent's current pos as key, their agent name (idx) as value
        for agentIdx in self.all_agents.keys():
            self.all_agents[agentIdx].pos = custom_agent_data[agentIdx][0:2]
            self.all_agents[agentIdx].ini_pos = custom_agent_data[agentIdx][0:2]
            self.all_agents[agentIdx].goal = [custom_agent_data[agentIdx][2:4]]
            self.all_agents[agentIdx].vel = custom_agent_data[agentIdx][4:6]
            # heading in rad, must be goal_pos-intruder_pos, and y2-y1, x2-x1
            self.all_agents[agentIdx].heading = math.atan2(self.all_agents[agentIdx].goal[0][1] -
                                                           self.all_agents[agentIdx].pos[1],
                                                           self.all_agents[agentIdx].goal[0][0] -
                                                           self.all_agents[agentIdx].pos[0])
            self.all_agents[agentIdx].observableSpace = self.current_observable_space(self.all_agents[agentIdx])
            cur_circle = Point(self.all_agents[agentIdx].pos[0],
                               self.all_agents[agentIdx].pos[1]).buffer(self.all_agents[agentIdx].protectiveBound,
                                                                        cap_style='round')
            agentRefer_dict[(self.all_agents[agentIdx].pos[0],
                             self.all_agents[agentIdx].pos[1])] = self.all_agents[agentIdx].agent_name

            # agentSTR_list.append(cur_circle)
            agentsCoor_list.append(self.all_agents[agentIdx].pos)

        self.cur_allAgentCoor_KD = KDTree(agentsCoor_list)

        # loop over all agent again to obtain each agent's detectable neighbor
        # second loop is required, because 1st loop is used to create the STR-tree of all agents
        # circle centre at their position
        for agentIdx, agent in self.all_agents.items():

            # get current agent's name in term of integer
            match = re.search(r'\d+(\.\d+)?', agent.agent_name)
            if match:
                agent_idx = int(match.group())
            else:
                agent_idx = None
                raise ValueError('No number found in string')

            # identify neighbors (use distance)
            # update the "surroundingNeighbor" attribute
            agent.surroundingNeighbor = self.get_current_agent_nei(agent, agentRefer_dict)
            # reset_world function we initialized both "surroundingNeighbor" and "pre_surroundingNeighbor" identically
            agent.pre_surroundingNeighbor = agent.surroundingNeighbor

            # populate the output stateV2 overall_state is a list of length equals to total number of agents
            agent_own = np.array([agent.pos[0], agent.pos[1], agent.goal[0][0], agent.goal[0][1], agent.vel[0], agent.vel[1]])
            overall_state.append(np.array([agent_own, agent.observableSpace, agent.surroundingNeighbor], dtype=object))

        if show:
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            matplotlib.use('TkAgg')
            fig, ax = plt.subplots(1, 1)
            for agentIdx, agent in self.all_agents.items():
                plt.plot(agent.pos[0], agent.pos[1], marker=MarkerStyle(">", fillstyle="right",
                                                                        transform=Affine2D().rotate_deg(
                                                                            math.degrees(agent.heading))), color='y')
                plt.text(agent.pos[0], agent.pos[1], agent.agent_name)
                # plot self_circle of the drone
                self_circle = Point(agent.pos[0], agent.pos[1]).buffer(agent.protectiveBound, cap_style='round')
                grid_mat_Scir = shapelypoly_to_matpoly(self_circle, False, 'k')
                ax.add_patch(grid_mat_Scir)

                # plot drone's detection range
                detec_circle = Point(agent.pos[0], agent.pos[1]).buffer(agent.detectionRange / 2, cap_style='round')
                detec_circle_mat = shapelypoly_to_matpoly(detec_circle, False, 'r')
                ax.add_patch(detec_circle_mat)

                ini = agent.pos
                for wp in agent.goal:
                    plt.plot(wp[0], wp[1], marker='*', color='y', markersize=10)
                    plt.plot([wp[0], ini[0]], [wp[1], ini[1]], '--', color='c')
                    ini = wp

            # draw occupied_poly
            for one_poly in self.world_map_2D_polyList[0][0]:
                one_poly_mat = shapelypoly_to_matpoly(one_poly, True, 'y', 'b')
                ax.add_patch(one_poly_mat)
            # draw non-occupied_poly
            for zero_poly in self.world_map_2D_polyList[0][1]:
                zero_poly_mat = shapelypoly_to_matpoly(zero_poly, False, 'y')
                # ax.add_patch(zero_poly_mat)

            # show building obstacles
            for poly in self.buildingPolygons:
                matp_poly = shapelypoly_to_matpoly(poly, False, 'red')  # the 3rd parameter is the edge color
                ax.add_patch(matp_poly)

            # plt.axvline(x=self.bound[0], c="green")
            # plt.axvline(x=self.bound[1], c="green")
            # plt.axhline(y=self.bound[2], c="green")
            # plt.axhline(y=self.bound[3], c="green")

            plt.xlabel("X axis")
            plt.ylabel("Y axis")
            plt.axis('equal')
            plt.show()

        return overall_state

    def get_current_agent_nei(self, cur_agent, agentRefer_dict):
        # identify neighbors (use distance)
        point_to_search = cur_agent.pos
        # subtract a small value to exclude point at exactly "search_distance"
        search_distance = (cur_agent.detectionRange / 2) + cur_agent.protectiveBound - 1e-6
        indices_from_KDtree = self.cur_allAgentCoor_KD.query_ball_point(point_to_search, search_distance)
        for possible_idx in indices_from_KDtree:
            # ensure the same agent will not goes into its surrounding neighbor attribute
            if not np.array_equal(self.cur_allAgentCoor_KD.data[possible_idx], point_to_search):
                other_agent_name = agentRefer_dict[tuple(self.cur_allAgentCoor_KD.data[possible_idx])]
                other_agent_idx = int(re.search(r'\d+(\.\d+)?', other_agent_name).group())
                cur_agent.surroundingNeighbor[other_agent_idx] = np.array([self.all_agents[other_agent_idx].pos[0],
                                                                           self.all_agents[other_agent_idx].pos[1],
                                                                           self.all_agents[other_agent_idx].vel[0],
                                                                           self.all_agents[other_agent_idx].vel[1],
                                                                           self.all_agents[other_agent_idx].goal[0][0],
                                                                           self.all_agents[other_agent_idx].goal[0][1]])
        return cur_agent.surroundingNeighbor

    def current_observable_space(self, cur_agent):
        occupied_building_val = 10
        occupied_drone_val = 50
        non_occupied_val = 1
        currentObservableState = []
        cur_hostPos_from_input = np.array([cur_agent.pos[0], cur_agent.pos[1]])
        t_x = cur_hostPos_from_input[0]
        t_y = cur_hostPos_from_input[1]
        polygonSet = []  # this polygonSet including the polygon that intersect with the "self_circle"
        self_circle_inter = []
        worldGrid_polyCombine = []
        # self.world_map_2D_polyList[0][0] is all grid=1, or list of occupied grids
        worldGrid_polyCombine.append(self.world_map_2D_polyList[0][0] + self.world_map_2D_polyList[0][1])
        world_STRtree = STRtree(worldGrid_polyCombine[0])
        detection_circle = Point(t_x, t_y).buffer(cur_agent.detectionRange / 2, cap_style='round')
        self_circle = Point(t_x, t_y).buffer(cur_agent.protectiveBound, cap_style='round')
        possible_matches = world_STRtree.query(detection_circle)

        for poly in world_STRtree.geometries.take(possible_matches):
            if detection_circle.intersects(poly):
                polygonSet.append(poly)
            if self_circle.intersects(poly):
                self_circle_inter.append(poly)

        # all detectable grids (not arranged)
        no_sorted_polySet = polygonSet

        # all detectable grids (arranged)
        sorted_polySet = sort_polygons(polygonSet)
        for poly in sorted_polySet:
            if self_circle.intersects(poly):
                currentObservableState.append(occupied_drone_val)
                continue
            if poly in self.world_map_2D_polyList[0][0]:
                currentObservableState.append(occupied_building_val)
            else:
                currentObservableState.append(non_occupied_val)
        # currently we using arranged polygonSet and 1D array
        return currentObservableState

    def get_actions_noCR(self, combine_state):
        outActions = {}
        noCR = 1
        for agent_idx, agent in self.all_agents.items():
            # heading in rad must be goal_pos-intruder_pos, and y2-y1, x2-x1
            agent.heading = math.atan2(agent.goal[0][1] - agent.pos[1],
                                       agent.goal[0][0] - agent.pos[0])
            agent.vel[0] = 10 * math.cos(agent.heading)
            agent.vel[1] = 10 * math.sin(agent.heading)
            outActions[agent_idx] = np.array([agent.vel[0], agent.vel[1]])
        return outActions, noCR

    def get_actions_NN(self, combine_state, eps):  # decentralized execution, only actor net is used here
        outActions = {}
        for agent_idx, agent in self.all_agents.items():
            # obtain the observation for each individual actor
            individual_obs = extract_individual_obs(combine_state, agent_idx)

            chosen_action = agent.choose_actions(individual_obs)
            # squeeze(0) is to remove the batch information
            # then convert to numpy
            chosen_action = chosen_action.squeeze(0).detach().numpy()
            chosen_action = chosen_action + self.OU_noise.noise()  # add noise for exploration

            # update current sigma used for the exploration
            self.OU_noise.sigma = self.OU_noise.largest_sigma * eps + (1 - eps) * self.OU_noise.smallest_sigma
            outActions[agent_idx] = chosen_action  # load to output dict

        return outActions

    def get_step_reward(self, current_ts):  # this is for individual drones, current_ts = current time step
        reward, done = [], []
        crash_penalty = -100
        reach_target = 1000
        potential_conflict_count = 0
        fixed_domino_reward = 1

        for drone_idx, drone_obj in self.all_agents.items():
            # re-initialize these two list for individual agents at each time step,this is to ensure collision
            # condition is reset for each agent at every time step
            collision_drones = []
            collide_building = 0
            pc_before, pc_after = [], []
            # we assume the maximum potential conflict the current drone could have at each time step is equals
            # to the total number of its neighbour at each time step
            pc_max_before = len(drone_obj.pre_surroundingNeighbor)
            pc_max_after = len(drone_obj.surroundingNeighbor)

            # calculate the deviation from the reference path after an action has been taken
            curPoint = Point(self.all_agents[drone_idx].pos)
            host_refline = LineString([self.all_agents[drone_idx].ini_pos, self.all_agents[drone_idx].goal[0]])
            cross_track_deviation = curPoint.distance(host_refline)  # deviation from the reference line, cross track error

            host_pass_line = LineString([self.all_agents[drone_idx].pre_pos, self.all_agents[drone_idx].pos])
            host_passed_volume = host_pass_line.buffer(self.all_agents[drone_idx].protectiveBound, cap_style='round')

            # neigh_keys is the drone_idx for current neighbors
            # loop through neighbors from previous time step
            for neigh_keys in self.all_agents[drone_idx].pre_surroundingNeighbor:
                # compute potential conflicts before and after the action for the current drone with its neighbours
                pc_before = compute_potential_conflict(pc_before, drone_obj.pre_pos, drone_obj.pre_vel,
                                                       drone_obj.protectiveBound, self.all_agents[neigh_keys].pre_pos,
                                                       self.all_agents[neigh_keys].pre_vel,
                                                       self.all_agents[neigh_keys].protectiveBound, neigh_keys,
                                                       current_ts)
            # loop through neighbors from current time step
            for neigh_keys in self.all_agents[drone_idx].surroundingNeighbor:
                # compute potential conflicts before and after the action for the current drone with its neighbours
                pc_after = compute_potential_conflict(pc_after, drone_obj.pos, drone_obj.vel,
                                                      drone_obj.protectiveBound, self.all_agents[neigh_keys].pos,
                                                      self.all_agents[neigh_keys].vel,
                                                      self.all_agents[neigh_keys].protectiveBound, neigh_keys,
                                                      current_ts)

                # check whether the current drone has collide with any surrounding neighbors due to current action
                neigh_pass_line = LineString([self.all_agents[neigh_keys].pre_pos, self.all_agents[neigh_keys].pos])
                neigh_passed_volume = neigh_pass_line.buffer(self.all_agents[neigh_keys].protectiveBound,
                                                             cap_style='round')
                if host_passed_volume.intersects(neigh_passed_volume):
                    print("drone_{} collide with drone_{} at time step {}".format(drone_idx, neigh_keys, current_ts))
                    collision_drones.append(neigh_keys)

            if pc_max_after == 0:  # upper bound in terms of value case for dominoTerm
                dominoTerm = fixed_domino_reward
            elif pc_max_before == 0:  # lower bound in terms of value case for dominoTerm
                dominoTerm = -1
            elif (len(pc_after)/pc_max_after) == 0:  # check if denominator of the dominoTerm equals to 0
                # if denominator equals to 0, meaning initial velocity is 0,
                # so is like initial condition, then we can just assign this dominoTerm as 0.
                dominoTerm = 0
            else:
                dominoTerm = ((len(pc_before)/pc_max_before) -
                              (len(pc_after)/pc_max_after)) / (len(pc_after)/pc_max_after)

            # check whether current actions leads to a collision with any buildings in the airspace
            allBuildingSTR = STRtree(self.world_map_2D_polyList[0][0])
            possiblePoly = allBuildingSTR.query(host_passed_volume)
            for element in possiblePoly:
                if allBuildingSTR.geometries.take(element).intersection(host_passed_volume):
                    collide_building = 1
                    print("drone_{} crash into building when moving from {} to {} at time step {}".format(drone_idx, self.all_agents[drone_idx].pre_pos, self.all_agents[drone_idx].pos, current_ts))
                    break

            tar_circle = Point(self.all_agents[drone_idx].goal[0]).buffer(1, cap_style='round')
            # when there is no intersection between two geometries, "RuntimeWarning" will appear
            # RuntimeWarning is, "invalid value encountered in intersection"
            goal_cur_intru_intersect = host_passed_volume.intersection(tar_circle)
            if not goal_cur_intru_intersect.is_empty:  # reached goal?
                print("drone_{} has reached its goal at time step {}".format(drone_idx, current_ts))
                done.append(True)
                reward.append(np.array(reach_target))
            # exceed bound or crash into buildings or crash with other neighbors
            elif (self.all_agents[drone_idx].pos[0] <= self.bound[0] or self.all_agents[drone_idx].pos[0] >= self.bound[1] or
                  self.all_agents[drone_idx].pos[1] <= self.bound[2] or self.all_agents[drone_idx].pos[1] >= self.bound[3] or
                  collide_building == 1 or len(collision_drones) > 0):
                reward.append(np.array(crash_penalty))
                done.append(True)
                if (collide_building == 0) or len(collision_drones) == 0:
                    print("drone_{} has crash into boundary at time step {}". format(drone_idx, current_ts))
            else:  # a normal step taken
                done.append(False)
                crossCoefficient = 1
                goalCoefficient = 6
                # cross track error term
                cross_track_error = (20 / ((cross_track_deviation * cross_track_deviation) / 200 + 1)) - 3.5
                # Distance between drone and its goal for two consecutive time step
                before_dist_hg = np.linalg.norm(drone_obj.pre_pos - drone_obj.goal[0])
                after_dist_hg = np.linalg.norm(drone_obj.pos - drone_obj.goal[0])  # distance to goal after action
                delta_hg = goalCoefficient * (before_dist_hg - after_dist_hg)
                # a small penalty for discourage the agent to stay in one single spot
                if (before_dist_hg - after_dist_hg) <= 2:
                    small_step_penalty = 50
                else:
                    small_step_penalty = 0
                # Domino term also use as an indicator for agent to avoid other drones. so no need to specifically
                # add an term to avoid surrounding drones
                step_reward = crossCoefficient*cross_track_error + delta_hg + dominoTerm - small_step_penalty
                # convert to arr
                step_reward = np.array(step_reward)
                reward.append(step_reward)

        return reward, done

    def step(self, actions, current_ts):
        next_combine_state = []
        agentCoorKD_list_update = []
        agentRefer_dict = {}  # A dictionary to use agent's current pos as key, their agent name (idx) as value
        # we use 4 here, because the time-step for the simulation is 0.5 sec.
        # hence, 4 here is equivalent to the acceleration of 2m/s^2
        coe_a = 4  # coe_a is the coefficient of action is 4.
        # based on the input stack of actions we propagate all agents forward
        for drone_idx, drone_act in actions.items():
            # let current neighbor become neighbor record before action
            self.all_agents[drone_idx].pre_surroundingNeighbor = self.all_agents[drone_idx].surroundingNeighbor
            # fill previous velocities
            self.all_agents[drone_idx].pre_vel = self.all_agents[drone_idx].vel
            ax, ay = drone_act[0, 0], drone_act[0, 1]
            # map output action from NN to actual range
            ax = map_range(ax, coe_a)
            ay = map_range(ay, coe_a)
            # check velocity limit
            curVelx = self.all_agents[drone_idx].vel[0] + ax * self.time_step
            curVely = self.all_agents[drone_idx].vel[1] + ay * self.time_step


            if np.linalg.norm([curVelx, curVely]) >= self.all_agents[drone_idx].maxSpeed:
                next_heading = math.atan2(curVely, curVelx)
                # update host velocity when chosen speed has exceed the max speed
                hvx = self.all_agents[drone_idx].maxSpeed * math.cos(next_heading)
                hvy = self.all_agents[drone_idx].maxSpeed * math.sin(next_heading)
                self.all_agents[drone_idx].vel = np.array([hvx, hvy])
            else:
                # update host velocity when max speed is not exceeded
                self.all_agents[drone_idx].vel = np.array([curVelx, curVely])

            #print("At time step {} the drone_{}'s output speed is {}".format(current_ts, drone_idx, np.linalg.norm(self.all_agents[drone_idx].vel)))

            # update the drone's position based on the update velocities
            self.all_agents[drone_idx].pre_pos = self.all_agents[drone_idx].pos
            delta_x = self.all_agents[drone_idx].vel[0] * self.time_step
            delta_y = self.all_agents[drone_idx].vel[1] * self.time_step
            self.all_agents[drone_idx].pos = np.array([self.all_agents[drone_idx].pos[0] + delta_x,
                                                       self.all_agents[drone_idx].pos[1] + delta_y])

            # cur_circle = Point(self.all_agents[drone_idx].pos[0],
            #                    self.all_agents[drone_idx].pos[1]).buffer(self.all_agents[drone_idx].protectiveBound,
            #                                                             cap_style='round')

            agentCoorKD_list_update.append(self.all_agents[drone_idx].pos)
            agentRefer_dict[(self.all_agents[drone_idx].pos[0],
                             self.all_agents[drone_idx].pos[1])] = self.all_agents[drone_idx].agent_name
        self.cur_allAgentCoor_KD = KDTree(agentCoorKD_list_update)  # update all agent coordinate KDtree

        # initiate for next state
        cur_ObsState = np.zeros((len(self.all_agents), 6))  # totalAgent * 6, 2D array
        cur_ObsGrids = []
        actor_obs = []

        # re-populate neighbors for all agents after action, and update states
        for agentIdx, agent in self.all_agents.items():
            # get current agent's name in term of integer
            match = re.search(r'\d+(\.\d+)?', agent.agent_name)
            if match:
                agent_idx = int(match.group())
            else:
                agent_idx = None
                raise ValueError('No number found in string')
            # update current agent's own state
            # agent_own = [agent.pos[0], agent.pos[1], agent.goal[0][0], agent.goal[0][1], agent.vel[0], agent.vel[1]]
            # cur_ObsState[agent_idx, :] = agent_own
            agent_own = np.array([agent.pos[0], agent.pos[1], agent.goal[0][0], agent.goal[0][1], agent.vel[0], agent.vel[1]])


            # update current agent's observable space state
            agent.observableSpace = self.current_observable_space(agent)
            #cur_ObsGrids.append(agent.observableSpace)

            # update the "surroundingNeighbor" attribute
            agent.surroundingNeighbor = self.get_current_agent_nei(agent, agentRefer_dict)
            #actor_obs.append(agent.surroundingNeighbor)

            # populate overall state
            next_combine_state.append(np.array([agent_own, agent.observableSpace, agent.surroundingNeighbor], dtype=object))



        # matplotlib.use('TkAgg')
        # plt.ion()
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.set_aspect('equal')
        #
        # for t in range(max_t):
        #     ax.set_xlim([self.bound[0], self.bound[1]])
        #     ax.set_ylim([self.bound[2], self.bound[3]])
        #     previous_time = deepcopy(self.global_time)
        #     cur_time = previous_time + self.time_step
        #     step_volume_collection = []
        #     agent_moving_vol = {}
        #     for agent_idx, agent in self.all_agents.items():
        #         previous_pos = deepcopy(agent.pos)
        #         dx = actions[agent_idx][0] * self.time_step
        #         dy = actions[agent_idx][1] * self.time_step
        #         agent.pos[0] = agent.pos[0] + dx
        #         agent.pos[1] = agent.pos[1] + dy
        #         cur_agent_passLine = LineString([(previous_pos[0], previous_pos[1]),
        #                                          (agent.pos[0], agent.pos[1])])
        #         cur_agent_passed_volume = cur_agent_passLine.buffer(agent.protectiveBound, cap_style='round')
        #         agent_moving_vol[agent_idx] = cur_agent_passed_volume
        #         step_volume_collection.append(cur_agent_passed_volume)
        #
        #         #plt.text(previous_pos[0], previous_pos[1], "{}, t={}".format(agent.agent_name, previous_time))
        #         matp_cur_volume = shapelypoly_to_matpoly(cur_agent_passed_volume, True, 'red', 'b')
        #         ax.add_patch(matp_cur_volume)
        #         plt.text(agent.pos[0], agent.pos[1], "{}".format(agent.agent_name))
        #
        #     step_volume_STR = STRtree(step_volume_collection)
        #
        #     # checking reach goal before the check collision. So that at the time step, when an agent reaches goal and
        #     # collide with other agent at the same time, it is consider as reaching destination instead of collision
        #
        #     collided_drone = []
        #     reached_drone = []
        #     for agentIdx_key, agent_passed_volume in agent_moving_vol.items():
        #         # check goal
        #         cur_drone_tar = Point(self.all_agents[agentIdx_key].goal[0][0],
        #                               self.all_agents[agentIdx_key].goal[0][1]).buffer(1, cap_style='round')
        #
        #         mat_cur_tar = shapelypoly_to_matpoly(cur_drone_tar, True, 'c', 'r')
        #         ax.add_patch(mat_cur_tar)
        #         plt.text(self.all_agents[agentIdx_key].goal[0][0],
        #                  self.all_agents[agentIdx_key].goal[0][1],
        #                  "{} goal".format(self.all_agents[agentIdx_key].agent_name))
        #
        #         if cur_drone_tar.intersects(agent_passed_volume):
        #             reached_drone.append(agentIdx_key)
        #             continue  # one drone reached its target no need to check any possible collision for this drone
        #
        #         # check collision
        #         possible_idx = step_volume_STR.query(agent_passed_volume)
        #         for other_agent_cir in step_volume_STR.geometries.take(possible_idx):
        #             if not other_agent_cir.equals(agent_passed_volume):
        #                 # record this volume only when not equals to itself.
        #                 collided_drone.append(agentIdx_key)
        #
        #     # if reached goal, remove the agent from the environment
        #     for i in reached_drone:
        #         del self.all_agents[i]
        #         print("agent_{} reached, it is removed from the environment".format(i))
        #     # Remove element in "collided_drone", such that these elements also present in "reached_drone"
        #     collided_drone = [x for x in collided_drone if x not in reached_drone]
        #     # remove possible duplicates in "collided_drone"
        #     collided_drone = list(set(collided_drone))
        #
        #     # if collide, remove any agents involved in the collision
        #     for i in collided_drone:
        #         del self.all_agents[i]
        #         print("removed agent_ {}, left {} agents".format(i, len(self.all_agents)))
        #     fig.canvas.draw()
        #     plt.show()
        #     time.sleep(2)
        #     fig.canvas.flush_events()
        #     ax.cla()
        return next_combine_state

    def central_learning(self, ReplayBuffer, batch_size, maxIntruNum, intruFeature):
        critic_losses, actor_losses = [], []
        cur_state, action, reward, next_state, done = ReplayBuffer.sample(batch_size, maxIntruNum, intruFeature, self.all_agents[0].max_grid_obs_dim)

        device = self.all_agents[0].actorNet.device

        # pre-process cur_state and next_state so that they can be used as input for every agent's critic network
        cur_state_pre_processed = preprocess_batch_for_critic_net(cur_state, batch_size)
        next_state_pre_processed = preprocess_batch_for_critic_net(next_state, batch_size)

        # load action, reward, done to tensor
        action = T.tensor(np.array(action), dtype=T.float).to(device)
        reward = T.tensor(np.array(reward), dtype=T.float).to(device)
        done = T.tensor(np.array(done)).to(device)

        # all these three different actions are needed to calculate the loss function
        all_agents_new_actions = []  # actions according to the target network for the new state
        all_agents_new_mu_actions = []  # actions according to the regular actor network for the current state
        old_agents_actions = []  # actions the agent actually tooks

        for agent_idx, agent in self.all_agents.items():
            # actions according to the target network for the new state
            next_own = T.tensor(next_state[agent_idx][0], dtype=T.float).to(device)
            next_grid = T.tensor(next_state[agent_idx][1], dtype=T.float).to(device)
            next_nei = T.tensor(next_state[agent_idx][2], dtype=T.float).to(device)
            agent_new_states = [next_own, next_grid, next_nei]

            new_pi = agent.target_actorNet.forward(agent_new_states)  # individual agent's target network

            all_agents_new_actions.append(new_pi)  # record actions generated by each agent's target network
            # actions according to the regular actor network for the current state
            cur_own = T.tensor(cur_state[agent_idx][0], dtype=T.float).to(device)
            cur_grid = T.tensor(cur_state[agent_idx][1], dtype=T.float).to(device)
            cur_nei = T.tensor(cur_state[agent_idx][2], dtype=T.float).to(device)
            mu_states = [cur_own, cur_grid, cur_nei]

            # using agent's predict network to generate action based off current states
            pi = agent.actorNet.forward(mu_states)

            all_agents_new_mu_actions.append(pi)
            # record actions the agent actually took
            old_agents_actions.append(action[agent_idx])

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)  # batch_size X agent_num X action dim
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions], dim=1)

        # These two losses are used to record overall losses for entire system at each learning step
        critic_losses = []
        actor_losses = []
        in_order_count = 0
        # handle cost function
        for agent_idx, agent in self.all_agents.items():
            # squeeze() will remove all dimensions with size 1
            # without squeeze() is 10x1x1, so is batch_number x 1 by 1 array.
            # using individual agent's critic prediction network
            # current Q estimate, shape is batch_size X 1
            critic_value = agent.criticNet.forward(cur_state_pre_processed, old_actions).squeeze(1)
            # get target Q value for each agent
            with T.no_grad():
                # next_state_pre_processed is arranged in a way that, always from 1st agent to last agent and ...
                # agent's idx is ignored, that's why they were stored in a list.
                # hence, "for agent_idx, agent in self.all_agents.items():" is just to loop through all agents in order
                # DO NOT use "agent_idx" as index, or else it will produce error, when add/remove agents are added.
                # First line for critic_value_prime does not involved any individual agent's attributes
                # because we using centralized critic, shape is batch_size X 1
                critic_value_prime = agent.target_criticNet.forward(next_state_pre_processed, new_actions).squeeze(1)
                critic_value_prime[done[in_order_count]] = 0.0
                target = reward[in_order_count] + agent.gamma * critic_value_prime
                in_order_count = in_order_count + 1

            # calculate critic loss for each agent
            critic_loss = F.mse_loss(critic_value, target)
            # optimization
            agent.criticNet.optimizer.zero_grad()
            critic_loss.backward()
            agent.criticNet.optimizer.step()

            # actor loss
            actor_loss = agent.criticNet.forward(cur_state_pre_processed, mu).squeeze()
            actor_loss = -T.mean(actor_loss)
            # actor optimization
            agent.actorNet.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            # actor_loss.backward()
            agent.actorNet.optimizer.step()

            agent.update_network_parameters()
            critic_losses.append(critic_loss)
            actor_losses.append(actor_loss)
        return critic_losses, actor_losses

    def save_model_actor_net(self, file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        # only actor net is required to be saved, because when evaluation, only actor network is required
        for agent_idx, agent_obj in self.all_agents.items():
            torch.save(agent_obj.actorNet.state_dict(), file_path + '/' +agent_obj.agent_name + 'actor_net')
            # torch.save(agent_obj.target_actorNet.state_dict(), file_path + '/' +agent_obj.agent_name + 'target_actor_net')
            # torch.save(agent_obj.criticNet.state_dict(), file_path + '/' +agent_obj.agent_name + 'critic_net')
            # torch.save(agent_obj.target_criticNet.state_dict(), file_path + '/' +agent_obj.agent_name + 'target_critic_net')






































