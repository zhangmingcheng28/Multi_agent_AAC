# -*- coding: utf-8 -*-
"""
@Time    : 3/2/2023 7:42 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""

from shapely.strtree import STRtree


class env_simulator:
    def __init__(self, world_map, building_polygons, grid_length, bound, occupiedGrid):
        self.world_map_2D = world_map
        self.gridlength = grid_length
        self.buildingPolygons = building_polygons
        self.bound = bound
        self.global_time = 0.0  # in sec
        self.time_step = 0.5  # in second as well
        self.zLevel_strTree = STRtree(occupiedGrid)
        self.all_agents = None

    def create_world(self, total_agentNum):
        self.all_agents = []
        for agent_i in range(total_agentNum):
            print("here")

            self.all_agents.append()







        self.reset_world()

    def reset_world(self):
        pass


