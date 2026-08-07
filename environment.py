import gym
from gym import Env
from gym.spaces import Discrete, Box

import numpy as np
import random
import os

from stable_baselines3 import PP0
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import roar_py_interface
import roar_py_carla
from submission import RoarCompetitionSolution
from infrastructure import RoarCompetitionAgentWrapper, ManualControlViewer
from typing import List, Type, Optional, Dict, Any
import carla
import asyncio
from LateralController import LatController
from ThrottleController import ThrottleController


class CustomEnv(Env):
    '''
    Custom Environment to train rl model for certain section.
    The goal for the rl implementation is to deploy specific models in specific sections of the track.
    Like a specialized contorller for each section.
    
    The environment is essentially rhe interface between the CARLA sim and the rl model.
    '''

    def __init__(self, world, section_start_index, section_end_index, spawn_waypoint_index):
        super().__init__()

        self.world = world

        self.section_start_index = section_start_index
        self.section_end_index = section_end_index
        self.spawn_waypoint_index = spawn_waypoint_index

        self.waypoints = None
        self.vehicle = None
        self.vehicle_wrapper = None

        self.camera_sensor = None
        self.location_sensor = None
        self.velocity_sensor = None
        self.rpy_sensor = None
        self.occupancy_map_sensor = None
        self.collision_sensor = None

        self.lat_controller = LatController()
        self.speed_controller = ThrottleController()

        self._respawn_location = None
        self._respawn_rpy = None
        self._last_vehicle_location = None

        self.current_waypoint_index = spawn_waypoint_index
        self.previous_progress = float(spawn_waypoint_index)

        self.previous_lookahead_action = 0.0
        self.previous_offset_action = 0.0

        self.episode_steps = 0
        self.max_episode_steps = 8000

        self.collision_threshold = 100.0
        self.collision_penalty = 200.0

        self.min_lookahead = 5
        self.max_lookahead = 40
        self.max_offset_meters = 2.0

        self.action_space = Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = ...

        self.is_setup = False
        self.is_closed = False

    async def setup(self) -> None:
        """
        Spawn the vehicle, attach sensors, settle the simulation,
        receive the first sensor observations, and initialize the race.

        Called only once before training.
        """

        if self.is_setup:
            return

        self.waypoints = list(self.world.maneuverable_waypoints)

        # Spawn the vehicle and attach sensors
        starting_waypoint = self.waypoints[0]

        vehicle = self.world.spawn_vehicle(
            "vehicle.tesla.model3",
            starting_waypoint.location
            + np.array([0.0, 0.0, 1.0]),
            starting_waypoint.roll_pitch_yaw,
            True,
        )

        self.vehicle = vehicle
        self.vehicle_wrapper = RoarCompetitionAgentWrapper(vehicle)

        self._attach_sensors()
        vehicle = self.vehicle

        self.camera_sensor = vehicle.attach_camera_sensor(
            roar_py_interface.RoarPyCameraSensorDataRGB,
            np.array(
                [
                    -2.0 * vehicle.bounding_box.extent[0],
                    0.0,
                    3.0 * vehicle.bounding_box.extent[2],
                ]
            ),
            np.array(
                [
                    0.0,
                    10.0 / 180.0 * np.pi,
                    0.0,
                ]
            ),
            image_width=1024,
            image_height=768,
        )

        self.location_sensor = (vehicle.attach_location_in_world_sensor())
        self.velocity_sensor = (vehicle.attach_velocimeter_sensor())
        self.rpy_sensor = (vehicle.attach_roll_pitch_yaw_sensor())
        self.occupancy_map_sensor = (vehicle.attach_occupancy_map_sensor(50,50,2.0,2.0,))
        self.collision_sensor = (vehicle.attach_collision_sensor(np.zeros(3),np.zeros(3),))

        sensors = {
            "camera": self.camera_sensor,
            "location": self.location_sensor,
            "velocity": self.velocity_sensor,
            "rotation": self.rpy_sensor,
            "occupancy map": self.occupancy_map_sensor,
            "collision": self.collision_sensor,
        }

        # Allow the spawned vehicle and sensors to initialize.
        for _ in range(20):
            await self.world.step()

        await self.vehicle.receive_observation()

        self.initialize_race()

        self.previous_progress = float(self.furthest_waypoint_index)
        self.previous_offset = 0.0
        self.previous_target_speed = self.min_target_speed
        self.episode_steps = 0

        self.is_setup = True

    def get_observation(self) -> np.ndarray:
        pass

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self):
        pass

    def initialize_race(self):
        self._last_vehicle_location = self.vehicle.get_3d_location()
        vehicle_location = self._last_vehicle_location
        closest_waypoint_dist = np.inf
        closest_waypoint_idx = 0
        for i,waypoint in enumerate(self.waypoints):
            waypoint_dist = np.linalg.norm(vehicle_location - waypoint.location)
            if waypoint_dist < closest_waypoint_dist:
                closest_waypoint_dist = waypoint_dist
                closest_waypoint_idx = i
        self.waypoints = self.waypoints[closest_waypoint_idx+1:] + self.waypoints[:closest_waypoint_idx+1]
        self.furthest_waypoints_index = 0
        print(f"total length: {len(self.waypoints)}")
        self._respawn_location = self._last_vehicle_location.copy()
        self._respawn_rpy = self.vehicle.get_roll_pitch_yaw().copy()