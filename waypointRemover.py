import matplotlib.pyplot as plt
import numpy as np
import roar_py_interface
from typing import List
from matplotlib.backend_bases import MouseButton

waypoints = roar_py_interface.RoarPyWaypoint.load_waypoint_list(
    np.load("competition_code\\waypoints\\modifiedWaypoints2.npz")
)
newWaypoints = []

for i in range(2660):
    newWaypoints.append(waypoints[i])

np.savez_compressed(
    "modifiedWaypoints.npz",
    **roar_py_interface.RoarPyWaypoint.save_waypoint_list(newWaypoints),
)
