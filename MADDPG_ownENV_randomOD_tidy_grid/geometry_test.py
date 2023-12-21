# -*- coding: utf-8 -*-
"""
@Time    : 8/11/2023 3:12 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
from shapely.strtree import STRtree
from shapely.geometry import LineString, Point, Polygon
from scipy.spatial import KDTree
import random
from copy import deepcopy
from agent_MADDPGv3 import Agent
import pandas as pd
import math
import numpy as np
import os
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from shapely.affinity import scale
import matplotlib.pyplot as plt
import matplotlib
import re
import time
from Utilities_own_MADDPGv3 import *
import torch as T
import torch
import torch.nn.functional as F
import torch.nn as nn

goal = Point(np.array([536, 356])).buffer(1, cap_style='round')
pre_pos = np.array([530.81, 353.08])
cur_pos = np.array([534.12, 355.86])