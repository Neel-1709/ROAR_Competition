import roar_py_carla
import roar_py_interface
import carla
import numpy as np
import asyncio
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
import transforms3d as tr3d


async def main():
    carla_client = carla.Client("localhost", 2000)
    carla_client.set_timeout(15.0)
    roar_py_instance = roar_py_carla.RoarPyCarlaInstance(carla_client)

    carla_world = roar_py_instance.world
    carla_world.set_asynchronous(True)
    carla_world.set_control_steps(0.00, 0.005)

    print("Map Name", carla_world.map_name)
    waypoints = roar_py_instance.world.maneuverable_waypoints
    spawn_points = roar_py_instance.world.spawn_points
    roar_py_instance.close()
    np.savez_compressed(
        f"{carla_world.map_name} Original Waypoints.npz",
        **roar_py_interface.RoarPyWaypoint.save_waypoint_list(waypoints),
    )


if __name__ == "__main__":
    asyncio.run(main())
