# -*- coding: utf-8 -*-
"""
@Time    : 3/2/2023 7:42 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""

from shapely.strtree import STRtree
from shapely.geometry import LineString, Point, Polygon
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
from Utilities_V1 import sort_polygons, shapelypoly_to_matpoly, extract_individual_obs


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
        self.cur_allAgent_STR = None

    def create_world(self, total_agentNum, critic_obs, actor_obs, n_actions, actorNet_lr, criticNet_lr, gamma, tau):
        self.all_agents = {}
        for agent_i in range(total_agentNum):
            agent = Agent(actor_obs, critic_obs, n_actions, agent_i, total_agentNum, actorNet_lr, criticNet_lr, gamma, tau)
            self.all_agents[agent_i] = agent
        global_state = self.reset_world(show=0)

    def reset_world(self, show):  # set initialize position and observation for all agents
        self.global_time = 0.0
        self.time_step = 0.5
        # prepare for output states
        cur_ObsState = np.zeros((len(self.all_agents), 6))  # totalAgent * 6, 2D array
        overall_state = []
        cur_ObsGrids = []
        actor_obs = []

        #  custom agent position
        # x-bound: [0, 1800), y-bound: [0, 1300)
        # read the Excel file into a pandas dataframe
        # df = pd.read_excel(r'F:\githubClone\Multi_agent_AAC\MA_ver1\fixedDrone.xlsx')
        df = pd.read_excel(r'D:\Multi_agent_AAC\MA_ver1\fixedDrone.xlsx')
        # convert the dataframe to a NumPy array
        custom_agent_data = np.array(df)
        custom_agent_data = custom_agent_data.astype(float)
        agentSTR_list = []  # for store all agents as circle polygon
        agentRefer_dict = {}  # A dictionary to use agent's current pos as key, their agent name (idx) as value
        for agentIdx in self.all_agents.keys():

            self.all_agents[agentIdx].pos = custom_agent_data[agentIdx][0:2]
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

            agentSTR_list.append(cur_circle)
        agent_cirSTR = STRtree(agentSTR_list)
        # loop over all agent again to obtain each agent's detectable neighbor
        # 2 second loop is required, because 1st loop is used to create the STR-tree of all agents
        # circle centre at their position
        for agentIdx, agent in self.all_agents.items():

            # get current agent's name in term of integer
            match = re.search(r'\d+(\.\d+)?', agent.agent_name)
            if match:
                agent_idx = int(match.group())
            else:
                agent_idx = None
                raise ValueError('No number found in string')

            # identify neighbors
            agent_cur_cir = Point(agent.pos[0], agent.pos[1]).buffer(agent.detectionRange/2, cap_style='round')
            possible_matches = agent_cirSTR.query(agent_cur_cir)
            for other_agent_cir in agent_cirSTR.geometries.take(possible_matches):
                if agent_cur_cir.intersects(other_agent_cir):
                    polyCen = (round(other_agent_cir.centroid.x, 1), round(other_agent_cir.centroid.y, 1))
                    # get the other agent's idx from the dictionary
                    other_agent_name = agentRefer_dict.get(polyCen)
                    if other_agent_name != agent.agent_name:
                        # get other_agent's idx in integer
                        other_agent_idx = int(re.search(r'\d+(\.\d+)?', other_agent_name).group())
                        agent.surroundingNeighbor[other_agent_idx] = np.array([self.all_agents[other_agent_idx].pos[0],
                                                                            self.all_agents[other_agent_idx].pos[1],
                                                                            self.all_agents[other_agent_idx].vel[0],
                                                                            self.all_agents[other_agent_idx].vel[1]])
            # end of identify neighbors

            # populate the output states
            agent_own = [agent.pos[0], agent.pos[1], agent.goal[0][0], agent.goal[0][1], agent.vel[0], agent.vel[1]]
            cur_ObsState[agent_idx, :] = agent_own
            cur_ObsGrids.append(agent.observableSpace)
            actor_obs.append(agent.surroundingNeighbor)
            overall_state = [cur_ObsState, cur_ObsGrids, actor_obs]

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
                self_circle = Point(agent.pos[0], agent.pos[1]).buffer(2.5, cap_style='round')
                grid_mat_Scir = shapelypoly_to_matpoly(self_circle, False, 'k')
                ax.add_patch(grid_mat_Scir)

                # plot drone's detection range
                detec_circle = Point(agent.pos[0], agent.pos[1]).buffer(30/2, cap_style='round')
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

    def get_actions_NN(self, combine_state):  # decentralized execution, only actor net is used here
        outActions = {}
        for agent_idx, agent in self.all_agents.items():
            # obtain the observation for each individual actor
            individual_obs = extract_individual_obs(combine_state, agent_idx)
            agent.choose_actions(individual_obs)
        print("all agent out action done")




    def step(self, noCR, actions, max_t):
        next_combine_state, reward, drone = 1, 1, 1
        matplotlib.use('TkAgg')
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')

        for t in range(max_t):
            ax.set_xlim([self.bound[0], self.bound[1]])
            ax.set_ylim([self.bound[2], self.bound[3]])
            previous_time = deepcopy(self.global_time)
            cur_time = previous_time + self.time_step
            if noCR == 1:
                step_volume_collection = []
                agent_moving_vol = {}
                for agent_idx, agent in self.all_agents.items():
                    previous_pos = deepcopy(agent.pos)
                    dx = actions[agent_idx][0] * self.time_step
                    dy = actions[agent_idx][1] * self.time_step
                    agent.pos[0] = agent.pos[0] + dx
                    agent.pos[1] = agent.pos[1] + dy
                    cur_agent_passLine = LineString([(previous_pos[0], previous_pos[1]),
                                                     (agent.pos[0], agent.pos[1])])
                    cur_agent_passed_volume = cur_agent_passLine.buffer(agent.protectiveBound, cap_style='round')
                    agent_moving_vol[agent_idx] = cur_agent_passed_volume
                    step_volume_collection.append(cur_agent_passed_volume)

                    #plt.text(previous_pos[0], previous_pos[1], "{}, t={}".format(agent.agent_name, previous_time))
                    matp_cur_volume = shapelypoly_to_matpoly(cur_agent_passed_volume, True, 'red', 'b')
                    ax.add_patch(matp_cur_volume)
                    plt.text(agent.pos[0], agent.pos[1], "{}".format(agent.agent_name))

                step_volume_STR = STRtree(step_volume_collection)

                # checking reach goal before the check collision. So that at the time step, when an agent reaches goal and
                # collide with other agent at the same time, it is consider as reaching destination instead of collision

                collided_drone = []
                reached_drone = []
                for agentIdx_key, agent_passed_volume in agent_moving_vol.items():
                    # check goal
                    cur_drone_tar = Point(self.all_agents[agentIdx_key].goal[0][0],
                                          self.all_agents[agentIdx_key].goal[0][1]).buffer(1, cap_style='round')

                    mat_cur_tar = shapelypoly_to_matpoly(cur_drone_tar, True, 'c', 'r')
                    ax.add_patch(mat_cur_tar)
                    plt.text(self.all_agents[agentIdx_key].goal[0][0],
                             self.all_agents[agentIdx_key].goal[0][1],
                             "{} goal".format(self.all_agents[agentIdx_key].agent_name))

                    if cur_drone_tar.intersects(agent_passed_volume):
                        reached_drone.append(agentIdx_key)
                        continue  # one drone reached its target no need to check any possible collision for this drone

                    # check collision
                    possible_idx = step_volume_STR.query(agent_passed_volume)
                    for other_agent_cir in step_volume_STR.geometries.take(possible_idx):
                        if not other_agent_cir.equals(agent_passed_volume):
                            # record this volume only when not equals to itself.
                            collided_drone.append(agentIdx_key)

                # if reached goal, remove the agent from the environment
                for i in reached_drone:
                    del self.all_agents[i]
                    print("agent_{} reached, it is removed from the environment".format(i))
                # Remove element in "collided_drone", such that these elements also present in "reached_drone"
                collided_drone = [x for x in collided_drone if x not in reached_drone]
                # remove possible duplicates in "collided_drone"
                collided_drone = list(set(collided_drone))

                # if collide, remove any agents involved in the collision
                for i in collided_drone:
                    del self.all_agents[i]
                    print("removed agent_ {}, left {} agents".format(i, len(self.all_agents)))

                fig.canvas.draw()
                plt.show()
                time.sleep(2)
                fig.canvas.flush_events()
                ax.cla()

        return next_combine_state, reward, drone

































