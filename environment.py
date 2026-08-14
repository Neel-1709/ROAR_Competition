import copy

from SectionStats import SectionStats
from SectionStats import filter_waypoints
import gymnasium as gym
from gymnasium import Env, spaces


import numpy as np
import random
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import roar_py_interface
import roar_py_carla
from submission import RoarCompetitionSolution, filter_waypoints
from infrastructure import RoarCompetitionAgentWrapper, ManualControlViewer
from typing import List, Type, Optional, Dict, Any
import carla
import asyncio
import threading
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

        # Lookahead distance and lateral offset
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.previous_action = np.zeros(2, dtype=np.float32)

        self.observation_space = self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(87,),
            dtype=np.float32,
        )

        self.previous_observation = np.zeros(87, dtype=np.float32)

        self.is_setup = False
        self.is_closed = False

        self.previous_yaw = None

        self.old_progress = 0.0

        # Make PPO synchronous gym environment compatible with asynchronous CARLA world
        self.async_loop = asyncio.new_event_loop()
        self.async_thread = threading.Thread(target=self.async_loop.run_forever, daemon=True)
        self.async_thread.start()

    def setup(self):
        if self.is_setup:
            return

        future = asyncio.run_coroutine_threadsafe(
            self.setup_async(),
            self.async_loop
        )

        future.result()
    
    async def setup_async(self) -> None:
        """
        Spawn the vehicle, attach sensors, settle the simulation,
        receive the first sensor observations, and initialize the race.

        Called only once before training.
        """

        if self.is_setup:
            return
        # Spawn the vehicle and attach sensors

        world_waypoints = self.world.maneuverable_waypoints
        spawn_wp = world_waypoints[0]

        self.vehicle = self.world.spawn_vehicle(
            "vehicle.tesla.model3",
            spawn_wp.location + np.array([0.0, 0.0, 1.0]),
            spawn_wp.roll_pitch_yaw,
            True,
        )

        if self.vehicle is None:
            raise RuntimeError("Vehicle failed to spawn at competition start")
        
        vehicle = self.vehicle

        starting_waypoint = self.world.maneuverable_waypoints[0]

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

        assert self.camera_sensor is not None
        assert self.location_sensor is not None
        assert self.velocity_sensor is not None
        assert self.rpy_sensor is not None
        assert self.occupancy_map_sensor is not None
        assert self.collision_sensor is not None

        self.vehicle_wrapper = RoarCompetitionAgentWrapper(vehicle)

        # Allow the spawned vehicle and sensors to initialize.
        for _ in range(20):
            await self.world.step()

        await self.vehicle.receive_observation()

        self.maneuverable_waypoints = (
            roar_py_interface.RoarPyWaypoint.load_waypoint_list(
                np.load(f"{os.path.dirname(__file__)}\\waypoints\\waypointsPrimary.npz")
            )[35:]
        )

        self.path_yaw = []

        for i in range(len(self.maneuverable_waypoints)):
            current_wp = self.maneuverable_waypoints[i]
            next_wp = self.maneuverable_waypoints[(i + 1) % len(self.maneuverable_waypoints)]

            dx = next_wp.location[0] - current_wp.location[0]
            dy = next_wp.location[1] - current_wp.location[1]
            yaw = np.arctan2(dy, dx)
            self.path_yaw.append(yaw)

        self.path_yaw = np.array(self.path_yaw)

        vehicle_location = self.location_sensor.get_last_gym_observation()

        self.current_waypoint_index = filter_waypoints(vehicle_location, self.spawn_waypoint_index, self.maneuverable_waypoints)

        self._respawn_location = self.vehicle.get_3d_location().copy()
        self._respawn_rpy = self.vehicle.get_roll_pitch_yaw().copy()

        self.initialize_race()

        self.old_progress = self.calculate_section_progress()
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.previous_yaw = None
        self.episode_steps = 0

        self.is_setup = True

    def calculate_section_progress(self):
        total_possible_progress = self.section_end_index - self.section_start_index
        current_progress = self.current_waypoint_index - self.section_start_index
        section_progress = current_progress / total_possible_progress
        section_progress = np.clip(section_progress, 0.0, 1.0)
        return section_progress
    
    def calculate_new_section_progress(self):
        new_progress = self.calculate_section_progress()
        section_progress_change = new_progress - self.old_progress
        self.old_progress = new_progress
        return section_progress_change

    def normalize_angle(self, angle):
       return (angle + np.pi) % (2 * np.pi) - np.pi

    def get_observation(self) -> np.ndarray:
        '''
        Features included in the observation space:
            - Yaw Rate
            - Heading Error
            - Speed
            - Lateral Error
            - Relative x offset of future 40 waypoints
            - Relative y offset of future 40 waypoints
            - Section Progress
            - Previous Action
        '''
        observation = np.empty((87,), dtype=np.float32)
        # Yaw Rate
        vehicle_rpy = self.rpy_sensor.get_last_gym_observation()
        current_yaw = vehicle_rpy[2]

        if self.previous_yaw is None:
            yaw_rate = 0.0
        else:
            yaw_change = self.normalize_angle(current_yaw - self.previous_yaw)
            yaw_rate = yaw_change / 0.05

        self.previous_yaw = current_yaw
        normalized_yaw_rate = np.clip(yaw_rate / 65.0, -1.0, 1.0)
        observation[0] = normalized_yaw_rate

        # Heading Error
        path_yaw = self.path_yaw[self.current_waypoint_index]

        heading_error = path_yaw - vehicle_rpy[2]
        heading_error = self.normalize_angle(heading_error)

        normalized_heading_error = heading_error / np.pi
        normalized_heading_error = np.clip(normalized_heading_error, -1.0, 1.0)
        observation[1] = normalized_heading_error

        # Speed
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        vehicle_velocity_norm = np.linalg.norm(vehicle_velocity)
        normalized_speed = np.clip((vehicle_velocity_norm * 3.6) / 60.0, -1.0, 1.0)
        observation[2] = normalized_speed

        # Lateral Error
        vehicle_location = self.location_sensor.get_last_gym_observation()
        previous_wp = self.maneuverable_waypoints[self.current_waypoint_index - 1].location
        next_wp = self.maneuverable_waypoints[(self.current_waypoint_index + 1) % len(self.maneuverable_waypoints)].location
        path = next_wp - previous_wp
        lateral_error = (path[1]*vehicle_location[0] - path[0]*vehicle_location[1] + next_wp[0]*previous_wp[1] - next_wp[1]*previous_wp[0]) / np.sqrt((path[0]**2 + path[1]**2))
        normalized_lateral_error = np.clip(lateral_error / 3.0, -1.0, 1.0)
        observation[3] = normalized_lateral_error

        # x and y offsets of future waypoints
        normalized_x_offsets = []
        normalized_y_offsets = []

        for i in range(1,41):
            idx = (self.current_waypoint_index + i * 5) % len(self.maneuverable_waypoints)

            waypoint = self.maneuverable_waypoints[idx]
            dx = waypoint.location[0] - vehicle_location[0]
            dy = waypoint.location[1] - vehicle_location[1]

            distance = np.sqrt(dx**2 + dy**2)
            global_angle_to_wp = np.arctan2(dy, dx)
            local_angle_to_wp = global_angle_to_wp - vehicle_rpy[2]

            normalized_relative_angle = self.normalize_angle(local_angle_to_wp)

            x_offset = distance * np.cos(normalized_relative_angle)
            y_offset = distance * np.sin(normalized_relative_angle)

            normalized_x_offset = np.clip(x_offset / 400.0, -1.0, 1.0)
            normalized_y_offset = np.clip(y_offset / 300.0, -1.0, 1.0)
            normalized_x_offsets.append(normalized_x_offset)
            normalized_y_offsets.append(normalized_y_offset)

        observation[4:44] = normalized_x_offsets
        observation[44:84] = normalized_y_offsets

        # Section Progress
        section_progress = self.calculate_section_progress()
        observation[84] = section_progress

        # Previous Action
        previous_lookahead_action = self.previous_action[0]
        previous_offset_action = self.previous_action[1]
        observation[85] = previous_lookahead_action
        observation[86] = previous_offset_action

        return observation

    def step(self, action):
        '''
        1. Action converted to lookahead distance and lateral offset
        2. Apply action to the vehicle using the lateral and throttle controllers
        3. Advance the simulation for a certain number of steps/ticks
        4. Get next world step's sensor data
        5. Calculate reward based on the acquired data
        6. Store action and get new observation for the next step
        7. Return observation, reward, done, truncated, info
        '''

        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        vehicle_velocity_norm = np.linalg.norm(vehicle_velocity)
        current_speed_kmh = vehicle_velocity_norm * 3.6

        # 1. Action converted to lookahead distance and lateral offset
        lookahead_distance = (action[0] + 1.0) / 2.0 * (self.max_lookahead - self.min_lookahead) + self.min_lookahead
        lateral_offset = action[1] * self.max_offset_meters

        # 2. Apply action to the vehicle using the lateral and throttle controllers
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()

        waypoints_to_shift = copy.deepcopy(self.maneuverable_waypoints[(self.current_waypoint_index + 1) % len(self.maneuverable_waypoints):(self.section_end_index % len(self.maneuverable_waypoints))])
        future_waypoints = self.shift_waypoint_path(lateral_offset, lookahead_distance, waypoints_to_shift)
        modified_wp = future_waypoints[int(lookahead_distance)]
        
        steer_control, deubg = self.lat_controller.run(vehicle_location, vehicle_rotation, modified_wp.location, self.current_waypoint_index)
        throttle, brake, gear, speed_data, debug_str = self.speed_controller.run(future_waypoints, vehicle_location, current_speed_kmh, self.get_current_section(), future_waypoints)

        # Hand-tuned steering multipliers not included to allow PPO to learn the optimal steering behavior.
        steerMultiplier = round((current_speed_kmh + 0.001) / 120, 3)
        steer_value = np.clip(steer_control * steerMultiplier, -1, 1)

        control = {
            "throttle": np.clip(throttle, 0, 1),
            "steer": steer_value,
            "brake": np.clip(brake, 0, 1),
            "hand_brake": 0,
            "reverse": 0,
            "target_gear": gear,
        }

        # 3. Advance the simulation for a certain number of steps
        future = asyncio.run_coroutine_threadsafe(self.next_world_step(control), self.async_loop)
        # Wait for the coroutine to complete
        future.result()
        
        # 4. Get next world step's sensor data
        vehicle_location = self.location_sensor.get_last_gym_observation()
        self.current_waypoint_index = filter_waypoints(vehicle_location, self.current_waypoint_index, self.maneuverable_waypoints)
        new_progress = self.calculate_new_section_progress()

        # 5. Calculate reward based on the acquired data
        crashed = False
        collision = np.linalg.norm(self.collision_sensor.get_last_observation().impulse_normal)
        if collision > self.collision_threshold:
            crashed = True
            collision = 1
        else:
            collision = 0
        reward = new_progress * (self.section_start_index - self.section_end_index) * 10 - (collision * self.collision_penalty) - 1

        # 6. Store action and get new observation for the next step
        self.previous_action = action
        self.previous_observation = self.get_observation()

        self.episode_steps += 1

        terminated = False
        truncated = False
        reason = ""

        info = {
            "done": terminated,
            "truncated": truncated,
            "reason": "",
        }

        if crashed:
            terminated = True
            reason = "Vehicle crashed"
            reward -= 100.0
        elif self.episode_steps >= self.max_episode_steps:
            truncated = True
            reason = "Max episode steps reached"
        elif self.current_waypoint_index >= self.section_end_index:
            terminated = True
            reason = "Completed Section"
            reward += 100.0

        info = {
            "done": terminated,
            "truncated": truncated,
            "reason": reason,
        }

        return self.previous_observation, reward, terminated, truncated, info

    async def next_world_step(self, control):
        await self.vehicle.apply_action(control)
        await self.world.step()
        await self.vehicle.receive_observation()

    def get_current_section(self):
        vehicle_location = self.location_sensor.get_last_gym_observation()
        section_num = filter_waypoints(vehicle_location, self.current_waypoint_index, self.maneuverable_waypoints)
        return section_num

    def get_lateral_error(self):
        vehicle_location = self.location_sensor.get_last_gym_observation()
        previous_wp = self.maneuverable_waypoints[self.current_waypoint_index - 1].location
        next_wp = self.maneuverable_waypoints[(self.current_waypoint_index + 1) % len(self.maneuverable_waypoints)].location
        path = next_wp - previous_wp
        path_norm = np.linalg.norm(np.sqrt((path[0]**2 + path[1]**2)))
        if path_norm == 0:
            lateral_error = 0
        else:
            lateral_error = (path[1]*vehicle_location[0] - path[0]*vehicle_location[1] + next_wp[0]*previous_wp[1] - next_wp[1]*previous_wp[0]) / np.sqrt((path[0]**2 + path[1]**2))
        return lateral_error      
    
    def shift_waypoint_path(self, shift_amount, lookahead_distance, waypoints_to_shift):
        shifted_waypoints = copy.deepcopy(waypoints_to_shift)
        lookahead_idx = int(lookahead_distance)
        before_len = min(lookahead_idx, len(shifted_waypoints))

        current_offset = self.get_lateral_error()

        for i, wp in enumerate(shifted_waypoints):
            prev_wp = self.maneuverable_waypoints[((self.current_waypoint_index + 1 + i) % len(self.maneuverable_waypoints) - 1) % len(self.maneuverable_waypoints)]
            next_wp = self.maneuverable_waypoints[((self.current_waypoint_index + 1 + i) % len(self.maneuverable_waypoints) + 1) % len(self.maneuverable_waypoints)]

            path = next_wp.location - prev_wp.location

            angle = np.arctan2(path[1], path[0])
            angle = self.normalize_angle(angle)
            perpendicular_angle = angle + np.pi / 2.0

            # Before target waypoint, shift from current offset to target offset
            if i < before_len:
                shift_percentage = 1 / max(before_len - 1, 1)
                # Smoothstep function
                smoothstep = 3 * (i * shift_percentage)**2 - 2 * (i * shift_percentage)**3
                shift = current_offset + (shift_amount - current_offset) * smoothstep
            # After target waypoint, shift from target offset to original line
            else:
                after_len = (len(shifted_waypoints) - before_len)
                j = i - before_len
                shift_percentage = 1 / max(after_len - 1, 1)
                # Smoothstep function
                smoothstep = 3 * (j * shift_percentage)**2 - 2 * (j * shift_percentage)**3
                shift = shift_amount * (1 - smoothstep)

            wp.location[0] += shift * np.cos(perpendicular_angle)
            wp.location[1] += shift * np.sin(perpendicular_angle)

        return shifted_waypoints

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        future = asyncio.run_coroutine_threadsafe(
            self.reset_async(),
            self.async_loop
        )

        future.result()

        observation = self.get_observation()

        info = {}

        return observation, info

    async def reset_async(self):
        self.vehicle.set_transform(self._respawn_location, self._respawn_rpy)
        self.vehicle.set_linear_3d_velocity(np.zeros(3))
        self.vehicle.set_angular_velocity(np.zeros(3))
        self.current_waypoint_index = (self.spawn_waypoint_index)
        self.episode_steps = 0
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.previous_yaw = None
        self.lat_controller = LatController()
        self.speed_controller = ThrottleController()

        for _ in range(20):
            await self.world.step()

        await self.vehicle.receive_observation()

        self.old_progress = self.calculate_section_progress()
        self._last_vehicle_location = self.vehicle.get_3d_location().copy()

    def render(self):
        return None

    def initialize_race(self):
        self._last_vehicle_location = self.vehicle.get_3d_location()
        # self.current_waypoint_index = self.spawn_waypoint_index
        self._respawn_location = self._last_vehicle_location.copy()
        self._respawn_rpy = self.vehicle.get_roll_pitch_yaw().copy()

    def close(self):
        if self.is_closed:
            return

        self.is_closed = True

        if self.vehicle is not None:
            self.vehicle.close()
            self.vehicle = None

        if self.camera_sensor is not None:
            self.camera_sensor.close()
            self.camera_sensor = None

        if self.location_sensor is not None:
            self.location_sensor.close()
            self.location_sensor = None

        if self.velocity_sensor is not None:
            self.velocity_sensor.close()
            self.velocity_sensor = None

        if self.rpy_sensor is not None:
            self.rpy_sensor.close()
            self.rpy_sensor = None

        if self.occupancy_map_sensor is not None:
            self.occupancy_map_sensor.close()
            self.occupancy_map_sensor = None

        if self.collision_sensor is not None:
            self.collision_sensor.close()
            self.collision_sensor = None

        self.async_loop.call_soon_threadsafe(self.async_loop.stop)
        self.async_thread.join()