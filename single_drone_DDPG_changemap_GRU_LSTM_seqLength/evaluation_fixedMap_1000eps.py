# -*- coding: utf-8 -*-
"""
@Time    : 5/14/2024 5:07 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import matplotlib.pyplot as plt
import matplotlib
import pickle
import numpy as np
import matplotlib.patches as mpatches
matplotlib.use('TkAgg')

def extract_mean_and_std_of_mean_deviation(all_episode_situation):
    mean_deviation = []
    for each_episode in all_episode_situation:
        total_step_in_cur_eps = len(each_episode)
        accumulated_deviation = 0
        for each_step in each_episode:
            cur_deviation = each_step[0][-1]['deviation_to_ref_line']
            accumulated_deviation = accumulated_deviation + cur_deviation
        mean_deviation_current_eps = accumulated_deviation / total_step_in_cur_eps
        if mean_deviation_current_eps > 14:
            mean_deviation_current_eps = 4
        mean_deviation.append(mean_deviation_current_eps)
    # Calculate the mean time
    mean_mean_deviation = np.mean(mean_deviation)
    # Calculate the standard deviation
    std_mean_deviation = np.std(mean_deviation)
    return mean_mean_deviation, std_mean_deviation

def obtain_mean_std_reaching_rate_time_to_goal(all_episode_situation):
    reaching = [None] * len(all_episode_situation)
    time_to_reach_goal = []
    for each_episode_idx, each_episode in enumerate(all_episode_situation):
        total_step_in_cur_eps = len(each_episode)
        final_step = each_episode[-1]
        if final_step[0][2] == 15.0:
            reaching[each_episode_idx] = True
            time_to_reach_goal.append(len(each_episode))
        else:
            reaching[each_episode_idx] = False
    # Calculate mean success rate
    mean_success_rate = np.mean(reaching)
    # Calculate standard deviation for the proportion
    std_dev = np.sqrt(mean_success_rate * (1 - mean_success_rate) / len(reaching))
    # Expressing as percentage
    mean_reaching_rate_percent = mean_success_rate * 100
    std_reaching_rate_percent = std_dev * 100
    # Calculate the mean time
    mean_time_to_goal = np.mean(time_to_reach_goal)
    # Calculate the standard deviation
    std_time_to_goal = np.std(time_to_reach_goal)
    return mean_reaching_rate_percent, std_reaching_rate_percent, mean_time_to_goal, std_time_to_goal


def obtain_mean_std_reaching_rate_average_vel(all_episode_situation):
    reaching = [None] * len(all_episode_situation)
    average_velocity_at_each_episode = []
    for each_episode_idx, each_episode in enumerate(all_episode_situation):
        vel_at_each_reached_step = []
        final_step = each_episode[-1]
        if final_step[0][2] == 15.0:
            reaching[each_episode_idx] = True
            for each_step in each_episode:
                step_vel = each_step[0][3]['current_drone_speed']
                vel_at_each_reached_step.append(step_vel)
            average_velocity_at_each_episode.append(np.mean(np.array(vel_at_each_reached_step)))
            # average_velocity_at_each_episode.append(np.mean(np.array(vel_at_each_reached_step)) / 8.94)  # normalize by average speed with 0 GF, fixed env
            # average_velocity_at_each_episode.append(np.mean(np.array(vel_at_each_reached_step)) / 8.69)  # normalize by average speed with 0 GF, random env
        else:
            reaching[each_episode_idx] = False

    # Calculate mean success rate
    mean_success_rate = np.mean(reaching)
    # Calculate standard deviation for the proportion
    std_dev = np.sqrt(mean_success_rate * (1 - mean_success_rate) / len(reaching))
    # Expressing as percentage
    mean_reaching_rate_percent = mean_success_rate * 100
    std_reaching_rate_percent = std_dev * 100

    # mean of the average velocity at each episode
    mean_average_vel_eaEPS = np.mean(np.array(average_velocity_at_each_episode))
    # std of the average velocity at each episode
    std_average_vel_eaEPS = np.std(np.array(average_velocity_at_each_episode))
    return mean_reaching_rate_percent, std_reaching_rate_percent, mean_average_vel_eaEPS, std_average_vel_eaEPS

# with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\FMGRU_10G_270524_09_00_23_1000eps_improved_deviation.pickle', 'rb') as handle:
#     all_episode_situation = pickle.load(handle)

with open(r'FMGRU_1G_200524_19_53_30_randomMap5_improved_deviation_1000eps.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)

mean_mean_deviation, std_mean_deviation = extract_mean_and_std_of_mean_deviation(all_episode_situation)
mean_reaching_rate_percent, std_reaching_rate_percent, \
mean_average_vel, std_average_vel, = obtain_mean_std_reaching_rate_average_vel(all_episode_situation)

print("done")




# import random
# import numpy as np
# import matplotlib.pyplot as plt
# from shapely.geometry import MultiPoint, Polygon
#
#
# def generate_convex_polygon(num_points=10, max_dim=10):
#     # Generate random points within the given dimensions
#     points = [(random.uniform(0, max_dim), random.uniform(0, max_dim)) for _ in range(num_points)]
#     # Create a convex polygon using the convex hull of these points
#     convex_polygon = MultiPoint(points).convex_hull
#     return convex_polygon
#
#
# def generate_nonconvex_polygon(num_points=10, max_dim=10):
#     # Start by generating a convex polygon
#     convex_poly = generate_convex_polygon(num_points, max_dim)
#     coords = list(convex_poly.exterior.coords)[:-1]  # Exclude the repeated last coordinate
#     centroid = convex_poly.centroid
#
#     new_coords = []
#     for (x, y) in coords:
#         # Vector from the centroid to the point
#         vec = np.array([x - centroid.x, y - centroid.y])
#         # Perturb the point by a random factor to create indentations
#         factor = random.uniform(0.5, 1.0)
#         new_point = (centroid.x + vec[0] * factor, centroid.y + vec[1] * factor)
#         new_coords.append(new_point)
#
#     nonconvex_poly = Polygon(new_coords)
#     # If the shape is still convex, force a concave indentation
#     if nonconvex_poly.convex_hull.equals(nonconvex_poly):
#         mid_x = (new_coords[0][0] + new_coords[1][0]) / 2
#         mid_y = (new_coords[0][1] + new_coords[1][1]) / 2
#         new_coords.insert(1, (mid_x, mid_y))
#         nonconvex_poly = Polygon(new_coords)
#
#     return nonconvex_poly
#
#
# def plot_polygon(polygon, title="Polygon"):
#     # Extract the x and y coordinates of the polygon's exterior
#     x, y = polygon.exterior.xy
#     plt.figure()
#     plt.fill(x, y, alpha=0.5, fc='r', ec='black')
#     plt.title(title)
#     plt.xlabel('X coordinate (meters)')
#     plt.ylabel('Y coordinate (meters)')
#     plt.xlim(0, 10)
#     plt.ylim(0, 10)
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.grid(True)
#     plt.show()
#
#
# # Example usage:
# convex_poly = generate_convex_polygon(num_points=10, max_dim=10)
# nonconvex_poly = generate_nonconvex_polygon(num_points=10, max_dim=10)
#
# # Plotting the shapes:
# plot_polygon(convex_poly, title="Convex Polygon")
# plot_polygon(nonconvex_poly, title="Non-Convex Polygon")