# -*- coding: utf-8 -*-
"""
@Time    : 3/20/2025 3:13 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
from shapely.geometry import MultiPoint
from matplotlib.patches import Polygon as matPolygon
import torch as T
import random
from matplotlib.patches import Rectangle, Circle
import matplotlib.patches as patches
import numpy as np
import torch
import os
import matplotlib
import matplotlib.pyplot as plt
from shapely.strtree import STRtree
from shapely.geometry import LineString, Point, Polygon
import matplotlib.colors as colors
from shapely.ops import nearest_points
from matplotlib.transforms import Affine2D
import matplotlib.animation as animation
from openpyxl import load_workbook
from openpyxl import Workbook
from matplotlib.markers import MarkerStyle
import math
from matplotlib.lines import Line2D
import pickle
from matplotlib.legend_handler import HandlerPatch


# Custom handler for the square patch
class HandlerSquare(HandlerPatch):
    def __init__(self, square_size_factor):
        self.square_size_factor = square_size_factor
        super().__init__()

    def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
        square_size = self.square_size_factor * min(width, height)
        square = plt.Rectangle((x0 + (width - square_size) / 2, y0 + (height - square_size) / 2), square_size, square_size, transform=trans, color='blue', alpha=0.9)
        return [square]

# this is to remove the need for the package descrete
def shapelypoly_to_matpoly(ShapelyPolgon, inFill=False, Edgecolor='black', FcColor='blue'):
    xcoo, ycoo = ShapelyPolgon.exterior.coords.xy
    matPolyConverted = matPolygon(xy=list(zip(xcoo, ycoo)), fill=inFill, edgecolor=Edgecolor, facecolor=FcColor)
    return matPolyConverted


def grayscale_to_rgba(value):
    return (value, value, value, 1)


with open('FMGRU_exact_10G_270524_09_00_23_plot_with_time_no_spawn_refline.pickle', 'rb') as f:
    stored_result = pickle.load(f)
episode_situation_holder = stored_result[4]
random_map_idx = stored_result[5]
bc = stored_result[6]
geo_fence_spawn_step = stored_result[7]

if None in episode_situation_holder:
    episode_situation_holder = list(filter(lambda x: x is not None, episode_situation_holder))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
matplotlib.use('TkAgg')
fig, ax = plt.subplots(1, 1)
font_size = 15
# Define color thresholds
light_gray = 0.8  # Lightest gray (not white)
dark_gray = 0.2  # Darkest gray (almost black)

# plt.axvline(x=env.bound_collection[random_map_idx][0], c="green")
# plt.axvline(x=env.bound_collection[random_map_idx][1], c="green")
# plt.axhline(y=env.bound_collection[random_map_idx][2], c="green")
# plt.axhline(y=env.bound_collection[random_map_idx][3], c="green")


# draw occupied_poly
for one_poly in stored_result[0][random_map_idx][0][0]:
    one_poly_mat = shapelypoly_to_matpoly(one_poly, True, 'y', 'b')
    ax.add_patch(one_poly_mat)
# draw non-occupied_poly
for zero_poly in stored_result[0][random_map_idx][0][1]:
    zero_poly_mat = shapelypoly_to_matpoly(zero_poly, False, 'y')
    # ax.add_patch(zero_poly_mat)

# show building obstacles
for poly in stored_result[1]:
    matp_poly = shapelypoly_to_matpoly(poly, False, 'red')  # the 3rd parameter is the edge color
    ax.add_patch(matp_poly)

# show geo-fence
for geo_fence in stored_result[2]:
    fence_poly = shapelypoly_to_matpoly(geo_fence, False, 'red')  # the 3rd parameter is the edge color
    gf_x = geo_fence.centroid.x
    gf_y = geo_fence.centroid.y
    plt.text(gf_x, gf_y, f'{geo_fence_spawn_step}s')
    ax.add_patch(fence_poly)

for agentIdx, agent in stored_result[3].items():
    dx = agent.goal[0][0] - agent.ini_pos[0]
    dy = agent.goal[0][1] - agent.ini_pos[1]
    angle = math.atan2(dy, dx)
    heading = math.degrees(angle)

    plt.plot(agent.ini_pos[0], agent.ini_pos[1],
             marker=MarkerStyle(">",
                                fillstyle="right",
                                transform=Affine2D().rotate_deg(heading)),
             color='g', markersize=10, label='Origin')
    # plt.text(agent.ini_pos[0], agent.ini_pos[1], 'Origin')
    plt.plot(agent.goal[-1][0], agent.goal[-1][1], marker='*', color='r', markersize=10, label='Destination')
    # plt.text(agent.goal[-1][0], agent.goal[-1][1], 'Destination')

    # link individual drone's starting position with its goal
    ini = agent.ini_pos
    # for wp in agent.goal:
    ic = 0
    for wp in agent.ref_line.coords:
        # plt.plot(wp[0], wp[1], marker='*', color='y', markersize=10)
        if ic == 0:
            plt.plot([wp[0], ini[0]], [wp[1], ini[1]], '--', color='c', label='Reference Line')
        else:
            plt.plot([wp[0], ini[0]], [wp[1], ini[1]], '--', color='c')
        ic = ic + 1
        ini = wp
    trajectory = []
    which_to_see = -1
    for eps_time_step, ea_eps in enumerate(episode_situation_holder[which_to_see]):
        t = eps_time_step
        radius = 2.5
        gray_value = light_gray - t * (light_gray - dark_gray) / (len(episode_situation_holder[which_to_see]) - 1)
        color_value = grayscale_to_rgba(gray_value)
        x = ea_eps[0][0]
        y = ea_eps[0][1]
        trajectory.append([x, y])
        # ax.plot(x, y, 'o', color=plt.cm.gray(color_value), markersize=10)
        # circle = patches.Circle((x, y), 2.5, color=color_value, ec='black')
        circle = patches.Circle((x, y), radius, facecolor=color_value, edgecolor='white')
        ax.add_patch(circle)
        if eps_time_step == geo_fence_spawn_step:
            # ax.text(x+15, y + 0, f't={eps_time_step}s', fontsize=9, ha='center', va='center', color='black')  # for 5 GF
            ax.text(x, y + 3, f't={eps_time_step}s', fontsize=9, ha='center', va='center', color='black')  # for 10 GF
            special_circle = patches.Circle((x, y), radius, facecolor='red', edgecolor='white')
            ax.add_patch(special_circle)
        # Plot large circle with dotted outline
        # outline_circle = patches.Circle((x, y), 7.5, facecolor='none', edgecolor=color_value, linestyle='dotted')
        # ax.add_patch(outline_circle)
    # Draw lines between the points
    trajectory = np.array(trajectory)
    ax.plot(trajectory[:, 0], trajectory[:, 1], color='black', linestyle='-', linewidth=1)

# Adding titles and labels
ax.set_title('Drone Trajectory with 10 Geo-fences\n(No Spawn Near Reference))',fontsize=font_size)
ax.set_xlabel('N-S direction (m)',fontsize=font_size)
# ax.set_ylabel('E-W direction (m)',fontsize=font_size, labelpad=-8)  # use for random figure a
ax.set_ylabel('E-W direction (m)',fontsize=font_size)
# ax.yaxis.set_label_coords(-0.1, 0.5)    # Adjust as needed
# Customize tick size
ax.tick_params(axis='both', which='major', labelsize=font_size)
# Custom legend
legend_elements = [
    Line2D([0], [0], linestyle='None', marker=MarkerStyle(">", fillstyle="right"), color='g', label='Origin', markerfacecolor='g', markersize=10),
    Line2D([0], [0], marker='*', color='w', label='Destination', markerfacecolor='r', markersize=15, linestyle='None'),
    Line2D([0], [0], color='cyan', lw=2, linestyle='dotted', label='Reference Line'),
    Line2D([0], [0], linestyle='None', marker='o', color='red', label='Geo-fence', markerfacecolor='none', markersize=10),
    Line2D([0], [0], linestyle='None', marker='o', color='white', label='Drone', markerfacecolor='black',
           markersize=10, alpha=0.5),
    Line2D([0], [0], linestyle='None', marker='o', color='red', label='Geo-fence-spawn', markerfacecolor='red',
           markersize=10, alpha=0.5),
    Rectangle((0, 0), 1, 1, color='blue', alpha=0.9, label='Building Grids')  # Blue grid square for legend
]

# Add legend
# Variable to control the size of the square legend item
# square_size_factor = 0.9  # Adjust this value as needed
#
# ax.legend(handles=legend_elements, loc='upper left', fontsize=font_size-2, borderpad=0.1, handler_map={Rectangle: HandlerSquare(square_size_factor=square_size_factor)
# })

plt.xlim(bc[random_map_idx][0], bc[random_map_idx][1]-5)
plt.ylim(bc[random_map_idx][2], bc[random_map_idx][3])
# plt.savefig('drone_trajectory.png')
plt.show()
