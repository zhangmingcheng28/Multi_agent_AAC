# -*- coding: utf-8 -*-
"""
@Time    : 3/30/2023 1:44 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""

from scipy.spatial import KDTree

points = [[1,2], [3,4], [5,6], [7,8], [9,10]]
kdtree = KDTree(points)

distance = 3.0
point_to_search = [4,5]

# Find all points within distance of point_to_search
indices = kdtree.query_ball_point(point_to_search, distance)

# Retrieve the corresponding points
points_within_distance = [points[i] for i in indices]

print("done")