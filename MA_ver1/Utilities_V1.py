# -*- coding: utf-8 -*-
"""
@Time    : 3/13/2023 1:28 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
from matplotlib.patches import Polygon as matPolygon
import torch as T
import numpy as np


def sort_polygons(polygons):  # this sorting is left to right, but bottom to top. so, 0th is below 2nd. [[2,3],
    # [0,1]]
    boxes = [polygon.bounds for polygon in polygons]
    sorted_boxes = sorted(boxes, key=lambda box: (box[1], box[0]))
    sorted_polygons = [polygons[boxes.index(box)] for box in sorted_boxes]
    return sorted_polygons


# this is to remove the need for the package descrete
def shapelypoly_to_matpoly(ShapelyPolgon, inFill=False, Edgecolor='black', FcColor='blue'):
    xcoo, ycoo = ShapelyPolgon.exterior.coords.xy
    matPolyConverted = matPolygon(xy=list(zip(xcoo, ycoo)), fill=inFill, edgecolor=Edgecolor, facecolor=FcColor)
    return matPolyConverted


def extract_individual_obs(combine_state, agent_idx):
    individual_obs = []
    self_obs = combine_state[agent_idx][0]
    self_obs_grid = combine_state[agent_idx][1]
    self_surround = combine_state[agent_idx][2]
    individual_obs = [self_obs, self_obs_grid, self_surround]
    return individual_obs


def map_range(value, coe_a):
    # Calculate the normalized value
    out_min = -coe_a
    out_max = coe_a
    # tanh() has range of -1 to 1
    in_min = -1
    in_max = 1
    normalized = (value - in_min) / (in_max - in_min)
    # Map the normalized value to the output range
    mapped = out_min + (normalized * (out_max - out_min))
    # Return the mapped value
    return mapped


def compute_potential_conflict(pc_list, cur_drone_pos, cur_drone_vel, cur_drone_protRad, cur_neigh_pos, cur_neigh_vel,
                               cur_neigh_protRad, cur_neigh_idx, current_ts):
    minus_rel_dist_before = -1 * (cur_drone_pos - cur_neigh_pos)  # always current drone - neighbours
    rel_vel_before = (cur_drone_vel - cur_neigh_vel)
    rel_vel_SQnorm_before = np.square(np.linalg.norm(rel_vel_before))
    if (current_ts == 0) & (rel_vel_SQnorm_before == 0):
        # this if-else if to remove the runtimeWarning due to getting a value of nan
        # because we initialized the velocity for each drone as 0 at start of each episode.
        # Therefore, will have runtime warning
        pass
    else:

        t_cpa_before = np.dot(minus_rel_dist_before, rel_vel_before) / rel_vel_SQnorm_before
        d_cpa_before = np.linalg.norm(((cur_drone_pos - cur_neigh_pos) + (rel_vel_before * t_cpa_before)))
        if (t_cpa_before >= 1) and (d_cpa_before < (cur_drone_protRad + cur_neigh_protRad)):
            pc_list.append(cur_neigh_idx)
    return pc_list


def padding_list(max_grid_obs_dim, input_list):
    # padding actions
    tobePad_gridObs = list(np.zeros(max_grid_obs_dim - len(input_list), dtype=int))
    padded_gridObs = input_list + tobePad_gridObs
    return padded_gridObs


def preprocess_batch_for_critic_net(input_state, batch_size):
    critic_own_batched_cur_state = []  # batch_size X one_agent_feature * max_num_agents
    critic_grid_batched_cur_state = []  # batch_size X one_agent_feature * max_num_agents
    critic_neigh_batched_cur_state = []  # batch_size X one_agent_feature * max_num_agents
    for batch_idx in range(batch_size):
        critic_own_cur_state = []
        critic_own_grid_state = []
        critic_own_neigh_state = []
        for agent_cur in input_state:
            critic_own_cur_state.append(agent_cur[0][batch_idx, :])
            critic_own_grid_state.append(agent_cur[1][batch_idx, :])
            # for neigh, first: max_nei_num X single_nei_features flatten to 1D array
            flat_nei = agent_cur[2][batch_idx, :].flatten()  # default is flatten in a row.
            critic_own_neigh_state.append(flat_nei)
        critic_own_batched_cur_state.append(np.array(critic_own_cur_state).reshape((1, -1)))
        critic_grid_batched_cur_state.append(np.array(critic_own_grid_state).reshape((1, -1)))
        critic_neigh_batched_cur_state.append(np.array(critic_own_neigh_state).reshape((1, -1)))

    cur_state_pre_processed = [T.tensor(np.array(critic_own_batched_cur_state)),
                               T.tensor(np.array(critic_grid_batched_cur_state)),
                               T.tensor(np.array(critic_neigh_batched_cur_state))]
    return cur_state_pre_processed


def neighbour_preprocess(neighbor_info, neighbor_feature):
    if len(neighbor_info) == 0:
        zero_tensor = T.zeros((1, neighbor_feature))  # 1x6 zero vector
        # when actor is picking an action, we only use actor's own observation + own grid_observation
        # + neighbors observation, if there is no neighbor detected, we use all zero vector to represent
        actions = self.actorNet.forward([ownObs, onwGridObs, zero_tensor])
    else:
        # handle n x 6
        neigh_arr = np.zeros((len(self.surroundingNeighbor), ownObs.shape[1]))
        # # ----------------------------------------------------------------------- # #
        # # to do: surrounding neighbour arrange in a way nearest neighbor at 1st or last  # #
        # # ------------------------------------------------------------------------# #
        for neigh_obs_idx, dict_keys in enumerate(
                self.surroundingNeighbor):  # loop through the dictionary in order, top first
            neigh_arr[neigh_obs_idx, :] = self.surroundingNeighbor[dict_keys]
        neigh_Obs = T.from_numpy(neigh_arr).float().to(self.actorNet.device)
        actions = self.actorNet.forward([ownObs, onwGridObs, neigh_Obs])





