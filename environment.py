import asyncio
import copy
import json
import os
import threading

import gymnasium as gym
import numpy as np
import roar_py_interface
from gymnasium import Env, spaces

from infrastructure import RoarCompetitionAgentWrapper, ManualControlViewer
from LateralController import LatController
from ThrottleController import ThrottleController
from submission import RoarCompetitionSolution, filter_waypoints


class CustomEnv(Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        world,
        section_start_index,
        section_end_index,
        spawn_waypoint_index,
        tick_repeat,
        time_to_beat,
        section_number,
    ):
        super().__init__()

        self.world = world
        self.section_start_index = int(section_start_index)
        self.section_end_index = int(section_end_index)
        self.spawn_waypoint_index = int(spawn_waypoint_index)
        self.tick_repeat = int(tick_repeat)
        self.time_to_beat = float(time_to_beat)
        self.section_number = int(section_number)

        self.vehicle = None
        self.vehicle_wrapper = None
        self.camera_sensor = None
        self.location_sensor = None
        self.velocity_sensor = None
        self.rpy_sensor = None
        self.occupancy_map_sensor = None
        self.collision_sensor = None

        self.maneuverable_waypoints = None
        self.path_yaw = None
        self.current_waypoint_index = self.spawn_waypoint_index

        self.lat_controller = LatController()
        self.speed_controller = ThrottleController()

        self.previous_action = np.zeros(2, dtype=np.float32)
        self.previous_observation = np.zeros(87, dtype=np.float32)
        self.previous_yaw = None
        self.old_progress = 0.0
        self.episode_steps = 0
        self.max_episode_steps = 8000
        self.section_start_time = 0.0

        self.collision_threshold = 50.0
        self.collision_penalty = 150.0
        self.min_lookahead = 5.0
        self.max_lookahead = 40.0
        self.max_offset_meters = 1.8

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(87,),
            dtype=np.float32,
        )

        self.is_setup = False
        self.is_closed = False

        self.enable_visualization = True
        self.viewer = ManualControlViewer()

        self.async_loop = asyncio.new_event_loop()
        self.async_thread = threading.Thread(
            target=self.async_loop.run_forever,
            daemon=True,
        )
        self.async_thread.start()

        self._competition_respawn_location = None
        self._competition_respawn_rpy = None

    def setup(self):
        if self.is_setup:
            return
        future = asyncio.run_coroutine_threadsafe(
            self.setup_async(),
            self.async_loop,
        )
        future.result()

    async def setup_async(self):
        if self.is_setup:
            return

        world_waypoints = self.world.maneuverable_waypoints

        self.vehicle = self.world.spawn_vehicle(
            "vehicle.tesla.model3",
            world_waypoints[0].location + np.array([0.0, 0.0, 1.0]),
            world_waypoints[0].roll_pitch_yaw,
            True,
        )
        if self.vehicle is None:
            raise RuntimeError("Vehicle failed to spawn at competition start.")

        vehicle = self.vehicle

        self.camera_sensor = vehicle.attach_camera_sensor(
            roar_py_interface.RoarPyCameraSensorDataRGB,
            np.array([
                -2.0 * vehicle.bounding_box.extent[0],
                0.0,
                3.0 * vehicle.bounding_box.extent[2],
            ]),
            np.array([0.0, 10.0 / 180.0 * np.pi, 0.0]),
            image_width=1024,
            image_height=768,
        )
        self.location_sensor = vehicle.attach_location_in_world_sensor()
        self.velocity_sensor = vehicle.attach_velocimeter_sensor()
        self.rpy_sensor = vehicle.attach_roll_pitch_yaw_sensor()
        self.occupancy_map_sensor = vehicle.attach_occupancy_map_sensor(
            50, 50, 2.0, 2.0
        )
        self.collision_sensor = vehicle.attach_collision_sensor(np.zeros(3), np.zeros(3))

        assert self.camera_sensor is not None
        assert self.location_sensor is not None
        assert self.velocity_sensor is not None
        assert self.rpy_sensor is not None
        assert self.occupancy_map_sensor is not None
        assert self.collision_sensor is not None

        self.vehicle_wrapper = RoarCompetitionAgentWrapper(vehicle)

        for _ in range(20):
            await self.world.step()

        await self.vehicle.receive_observation()

        self._respawn_location = self.vehicle.get_3d_location().copy()
        self._respawn_rpy = self.vehicle.get_roll_pitch_yaw().copy()

        print(
            "SAVED RESPAWN:",
            self._respawn_location,
            self._respawn_rpy
        )

        self._competition_respawn_location = self.vehicle.get_3d_location().copy()
        self._competition_respawn_rpy = self.vehicle.get_roll_pitch_yaw().copy()
        self.maneuverable_waypoints = (
            roar_py_interface.RoarPyWaypoint.load_waypoint_list(
                np.load(
                    f"{os.path.dirname(__file__)}\\waypoints\\waypointsPrimary.npz"
                )
            )[35:]
        )

        self.path_yaw = np.empty(len(self.maneuverable_waypoints), dtype=np.float64)

        for i, current_wp in enumerate(self.maneuverable_waypoints):
            next_wp = self.maneuverable_waypoints[(i + 1) % len(self.maneuverable_waypoints)]
            dx = next_wp.location[0] - current_wp.location[0]
            dy = next_wp.location[1] - current_wp.location[1]
            self.path_yaw[i] = np.arctan2(dy, dx)

        self.current_waypoint_index = self.spawn_waypoint_index
        self.is_setup = True

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def _forward_distance(self, start_idx, end_idx):
        n = len(self.maneuverable_waypoints)
        return (end_idx - start_idx) % n

    def _section_length(self):
        return self._forward_distance(self.section_start_index, self.section_end_index)

    def calculate_section_progress(self):
        total = self._section_length()
        if total == 0:
            return 1.0

        progressed = self._forward_distance(self.section_start_index, self.current_waypoint_index)

        return float(np.clip(progressed / total, 0.0, 1.0))

    def calculate_new_section_progress(self):
        new_progress = self.calculate_section_progress()
        delta = new_progress - self.old_progress
        if delta < -0.5:
            delta = 0.0
        self.old_progress = new_progress
        return float(delta)

    def _section_completed(self):
        return self.calculate_section_progress() >= 1.0

    def _current_section_from_index(self, idx):
        if 322 <= idx < 557:
            return 1
        if 557 <= idx < 739:
            return 2
        if 739 <= idx < 1158:
            return 3
        if 1158 <= idx < 1317:
            return 4
        if 1317 <= idx < 1516:
            return 5
        if 1516 <= idx < 1881:
            return 6
        if 1881 <= idx < 1944:
            return 7
        if 1944 <= idx < 2359:
            return 8
        if 2359 <= idx < 2611:
            return 9
        return 0

    def _waypoints_until_section_end(self):
        n = len(self.maneuverable_waypoints)
        remaining = self._forward_distance(self.current_waypoint_index, self.section_end_index)
        if remaining == 0:
            return []

        return copy.deepcopy([
            self.maneuverable_waypoints[
                (self.current_waypoint_index + i) % n
            ]
            for i in range(1, remaining + 1)
        ])

    def get_observation(self):
        observation = np.empty((87,), dtype=np.float32)

        vehicle_rpy = self.rpy_sensor.get_last_gym_observation()
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()

        current_yaw = vehicle_rpy[2]
        if self.previous_yaw is None:
            yaw_rate = 0.0
        else:
            yaw_change = self.normalize_angle(current_yaw - self.previous_yaw)
            yaw_rate = yaw_change / 0.05

        self.previous_yaw = current_yaw
        observation[0] = np.clip(yaw_rate / 65.0, -1.0, 1.0)

        path_yaw = self.path_yaw[self.current_waypoint_index]
        heading_error = self.normalize_angle(path_yaw - current_yaw)
        observation[1] = np.clip(heading_error / np.pi, -1.0, 1.0)

        speed_kmh = np.linalg.norm(vehicle_velocity) * 3.6
        observation[2] = np.clip(speed_kmh / 60.0, -1.0, 1.0)

        previous_wp = self.maneuverable_waypoints[(self.current_waypoint_index - 1) % len(self.maneuverable_waypoints)].location
        next_wp = self.maneuverable_waypoints[(self.current_waypoint_index + 1) % len(self.maneuverable_waypoints)].location

        path = next_wp - previous_wp
        path_norm = np.sqrt(path[0] ** 2 + path[1] ** 2)

        if path_norm < 1e-8:
            lateral_error = 0.0
        else:
            lateral_error = (path[1] * vehicle_location[0] - path[0] * vehicle_location[1] + next_wp[0] * previous_wp[1] - next_wp[1] * previous_wp[0]) / path_norm

        observation[3] = np.clip(lateral_error / 3.0, -1.0, 1.0)

        x_offsets = []
        y_offsets = []

        for i in range(1, 41):
            idx = (self.current_waypoint_index + i * 5) % len(self.maneuverable_waypoints)

            waypoint = self.maneuverable_waypoints[idx]
            dx = waypoint.location[0] - vehicle_location[0]
            dy = waypoint.location[1] - vehicle_location[1]

            distance = np.sqrt(dx ** 2 + dy ** 2)
            global_angle = np.arctan2(dy, dx)
            local_angle = self.normalize_angle(global_angle - current_yaw)

            x_offsets.append(
                np.clip(
                    distance * np.cos(local_angle) / 400.0,
                    -1.0,
                    1.0,
                )
            )
            y_offsets.append(
                np.clip(
                    distance * np.sin(local_angle) / 300.0,
                    -1.0,
                    1.0,
                )
            )

        observation[4:44] = x_offsets
        observation[44:84] = y_offsets
        observation[84] = self.calculate_section_progress()
        observation[85:87] = self.previous_action

        return observation

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        lookahead_distance = (action[0] + 1.0) / 2.0 * (self.max_lookahead - self.min_lookahead) + self.min_lookahead
        lateral_offset = action[1] * self.max_offset_meters

        total_progress = 0.0
        collision = 0.0
        crashed = False
        completed = False
        ticks_executed = 0

        for _ in range(self.tick_repeat):
            vehicle_location = self.location_sensor.get_last_gym_observation()
            vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
            vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
            current_speed_kmh = np.linalg.norm(vehicle_velocity) * 3.6

            waypoints_to_shift = self._waypoints_until_section_end()
            if not waypoints_to_shift:
                completed = True
                break

            future_waypoints = self.shift_waypoint_path(
                lateral_offset,
                lookahead_distance,
                waypoints_to_shift,
            )

            target_local_idx = min(max(int(lookahead_distance) - 1, 0), len(future_waypoints) - 1)
            modified_wp = future_waypoints[target_local_idx]

            steer_control, _ = self.lat_controller.run(
                vehicle_location,
                vehicle_rotation,
                modified_wp.location,
                self.current_waypoint_index,
            )

            wp_len = len(self.maneuverable_waypoints)

            THROTTLE_HORIZON = 300

            throttle_waypoints = [
                self.maneuverable_waypoints[
                    (self.current_waypoint_index + i) % wp_len
                ]
                for i in range(1, THROTTLE_HORIZON + 1)
            ]

            additional_start = (
                self.current_waypoint_index - 9
            ) % wp_len

            additional_waypoints = [
                self.maneuverable_waypoints[
                    (additional_start + i) % wp_len
                ]
                for i in range(THROTTLE_HORIZON)
            ]

            throttle, brake, gear, _, _ = self.speed_controller.run(
                throttle_waypoints,
                vehicle_location,
                current_speed_kmh,
                self.section_number,
                additional_waypoints,
            )

            steer_multiplier = round((current_speed_kmh + 0.001) / 120, 3)

            steer_value = np.clip(steer_control * steer_multiplier, -1, 1)

            control = {
                "throttle": np.clip(throttle, 0, 1),
                "steer": steer_value,
                "brake": np.clip(brake, 0, 1),
                "hand_brake": 0,
                "reverse": 0,
                "target_gear": gear,
            }

            future = asyncio.run_coroutine_threadsafe(
                self.next_world_step(control),
                self.async_loop,
            )
            future.result()

            ticks_executed += 1

            if self.enable_visualization:
                self.render_camera()

            vehicle_location = self.location_sensor.get_last_gym_observation()

            self.current_waypoint_index = filter_waypoints(
                vehicle_location,
                self.current_waypoint_index,
                self.maneuverable_waypoints,
            )

            total_progress += self.calculate_new_section_progress()

            collision_impulse = np.linalg.norm(self.collision_sensor.get_last_observation().impulse_normal)

            if collision_impulse > self.collision_threshold:
                crashed = True
                collision = 1.0
                break

            if self._section_completed():
                completed = True
                break

        reward = 120 * total_progress - collision * self.collision_penalty - 0.10 * ticks_executed - 0.07 * float(np.sum((self.previous_action - action) ** 2))

        self.previous_action = action.copy()
        self.previous_observation = self.get_observation()
        self.episode_steps += 1

        section_time = self.world.last_tick_elapsed_seconds - self.section_start_time

        terminated = False
        truncated = False
        reason = ""

        if crashed:
            terminated = True
            reason = "Vehicle crashed"
        elif completed:
            terminated = True
            reason = "Completed Section"
            reward += 250.0

            reward += 180.0 * max(0.0, 22.25 - section_time)

        elif self.episode_steps >= self.max_episode_steps:
            truncated = True
            reason = "Max episode steps reached"

        info = {
            "reason": reason,
            "time": float(section_time),
            "progress": self.calculate_section_progress(),
            "crashed": crashed,
            "completed": completed,
        }

        return self.previous_observation, float(reward), terminated, truncated, info

    async def next_world_step(self, control):
        await self.vehicle.apply_action(control)
        await self.world.step()
        await self.vehicle.receive_observation()

    def get_lateral_error(self):
        vehicle_location = self.location_sensor.get_last_gym_observation()
        previous_wp = self.maneuverable_waypoints[(self.current_waypoint_index - 1) % len(self.maneuverable_waypoints)].location
        next_wp = self.maneuverable_waypoints[(self.current_waypoint_index + 1) % len(self.maneuverable_waypoints)].location

        path = next_wp - previous_wp
        path_norm = np.sqrt(path[0] ** 2 + path[1] ** 2)

        if path_norm < 1e-8:
            return 0.0

        return (path[1] * vehicle_location[0] - path[0] * vehicle_location[1] + next_wp[0] * previous_wp[1] - next_wp[1] * previous_wp[0]) / path_norm

    def shift_waypoint_path(self, shift_amount, lookahead_distance, waypoints_to_shift):
        shifted_waypoints = copy.deepcopy(waypoints_to_shift)
        before_len = min(int(lookahead_distance), len(shifted_waypoints))
        current_offset = self.get_lateral_error()

        for i, wp in enumerate(shifted_waypoints):
            track_idx = (self.current_waypoint_index + 1 + i) % len(self.maneuverable_waypoints)

            prev_wp = self.maneuverable_waypoints[(track_idx - 1) % len(self.maneuverable_waypoints)]
            next_wp = self.maneuverable_waypoints[(track_idx + 1) % len(self.maneuverable_waypoints)]

            path = next_wp.location - prev_wp.location
            angle = np.arctan2(path[1], path[0])
            perpendicular_angle = (self.normalize_angle(angle) + np.pi / 2.0)

            if i < before_len:
                fraction = i / max(before_len - 1, 1)
                smoothstep = (3 * fraction ** 2 - 2 * fraction ** 3)
                shift = current_offset + (shift_amount - current_offset) * smoothstep
            else:
                after_len = len(shifted_waypoints) - before_len
                j = i - before_len
                fraction = j / max(after_len - 1, 1)
                smoothstep = 3 * fraction ** 2 - 2 * fraction ** 3
                shift = shift_amount * (1.0 - smoothstep)

            wp.location[0] += shift * np.cos(perpendicular_angle)
            wp.location[1] += shift * np.sin(perpendicular_angle)

        return shifted_waypoints

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        future = asyncio.run_coroutine_threadsafe(
            self.reset_async(),
            self.async_loop,
        )
        future.result()

        self.episode_steps = 0
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.previous_yaw = None
        self.old_progress = self.calculate_section_progress()

        self.section_start_time = self.world.last_tick_elapsed_seconds

        observation = self.get_observation()
        return observation, {}

    async def reset_async(self):
        max_baseline_retries = 5

        with open("baseline_states.json", "r") as f:
            baseline_states = json.load(f)

        spawn_key = str(self.spawn_waypoint_index)

        if spawn_key not in baseline_states:
            raise RuntimeError(
                f"No baseline state recorded for waypoint "
                f"{self.spawn_waypoint_index}"
            )

        spawn_state = baseline_states[spawn_key]

        for attempt in range(max_baseline_retries):
            if self.vehicle is not None:
                self.vehicle.close()
                self.vehicle = None

            self.camera_sensor = None
            self.location_sensor = None
            self.velocity_sensor = None
            self.rpy_sensor = None
            self.occupancy_map_sensor = None
            self.collision_sensor = None
            self.vehicle_wrapper = None

            for _ in range(2):
                await self.world.step()

            world_waypoints = self.world.maneuverable_waypoints

            self.vehicle = self.world.spawn_vehicle(
                "vehicle.tesla.model3",
                world_waypoints[0].location
                + np.array([0.0, 0.0, 1.0]),
                world_waypoints[0].roll_pitch_yaw,
                True,
            )

            if self.vehicle is None:
                print(
                    f"Vehicle spawn failed on attempt "
                    f"{attempt + 1}/{max_baseline_retries}"
                )
                continue

            vehicle = self.vehicle

            self.camera_sensor = vehicle.attach_camera_sensor(
                roar_py_interface.RoarPyCameraSensorDataRGB,
                np.array([
                    -2.0 * vehicle.bounding_box.extent[0],
                    0.0,
                    3.0 * vehicle.bounding_box.extent[2],
                ]),
                np.array([
                    0.0,
                    10.0 / 180.0 * np.pi,
                    0.0,
                ]),
                image_width=1024,
                image_height=768,
            )

            self.location_sensor = vehicle.attach_location_in_world_sensor()
            self.velocity_sensor = vehicle.attach_velocimeter_sensor()
            self.rpy_sensor = vehicle.attach_roll_pitch_yaw_sensor()
            self.occupancy_map_sensor = vehicle.attach_occupancy_map_sensor(50, 50, 2.0, 2.0)
            self.collision_sensor = vehicle.attach_collision_sensor(np.zeros(3), np.zeros(3))

            self.vehicle_wrapper = RoarCompetitionAgentWrapper(vehicle)

            for _ in range(5):
                await self.world.step()

            spawn_location = np.array(spawn_state["location"], dtype=np.float64)
            spawn_rpy = np.array(spawn_state["rpy"], dtype=np.float64)
            spawn_velocity = np.array(spawn_state["linear_velocity"], dtype=np.float64)

            self.vehicle.set_transform(spawn_location, spawn_rpy)
            self.vehicle.set_linear_3d_velocity(spawn_velocity)
            self.vehicle.set_angular_velocity(np.zeros(3))

            neutral_control = {
                "throttle": 0.0,
                "steer": 0.0,
                "brake": 0.0,
                "hand_brake": 0,
                "reverse": 0,
                "target_gear": 1,
            }

            await self.vehicle.apply_action(neutral_control)

            await self.world.step()
            await self.vehicle.receive_observation()

            baseline = RoarCompetitionSolution(
                world_waypoints,
                self.vehicle_wrapper,
                self.camera_sensor,
                self.location_sensor,
                self.velocity_sensor,
                self.rpy_sensor,
                self.occupancy_map_sensor,
                self.collision_sensor,
            )

            await baseline.initialize()
            baseline.disable_waypoint_line = True

            baseline.current_waypoint_idx = self.spawn_waypoint_index
            self.current_waypoint_index = self.spawn_waypoint_index
            baseline.current_section = self._current_section_from_index(self.spawn_waypoint_index)
            baseline.previous_timing_section = baseline.current_section

            await baseline.step()
            await self.world.step()
            await self.vehicle.receive_observation()

            warmup_origin = self.spawn_waypoint_index

            warmup_target_distance = self._forward_distance(warmup_origin, self.section_start_index)

            warmup_steps = 0
            max_warmup_steps = 1000
            warmup_failed = False

            while True:
                current_distance = self._forward_distance(warmup_origin, baseline.current_waypoint_idx)

                if current_distance >= warmup_target_distance:
                    break

                await self.vehicle.receive_observation()

                if self.enable_visualization:
                    result = self.viewer.render(self.camera_sensor.get_last_observation())

                    if result is None:
                        raise RuntimeError("Viewer was closed.")

                collision_impulse = np.linalg.norm(self.collision_sensor.get_last_observation().impulse_normal)

                if collision_impulse > self.collision_threshold:
                    print(
                        f"Baseline warmup crashed on "
                        f"attempt {attempt + 1}/"
                        f"{max_baseline_retries}. "
                        f"Restarting snapshot..."
                    )

                    warmup_failed = True
                    break

                await baseline.step()
                await self.world.step()

                self.current_waypoint_index = baseline.current_waypoint_idx

                warmup_steps += 1

                if warmup_steps >= max_warmup_steps:
                    print(
                        f"Baseline warmup timed out on "
                        f"attempt {attempt + 1}/"
                        f"{max_baseline_retries}."
                    )

                    warmup_failed = True
                    break

            if warmup_failed:
                continue

            await self.vehicle.receive_observation()

            vehicle_location = self.location_sensor.get_last_gym_observation()

            self.current_waypoint_index = filter_waypoints(vehicle_location, baseline.current_waypoint_idx, self.maneuverable_waypoints)

            self.lat_controller = LatController()
            self.speed_controller = ThrottleController()

            self.episode_steps = 0

            self.previous_action = np.zeros(2, dtype=np.float32)

            self.previous_yaw = None

            self.old_progress = self.calculate_section_progress()

            print(
                f"Baseline warmup succeeded. "
                f"Snapshot={self.spawn_waypoint_index}, "
                f"PPO takeover="
                f"{self.current_waypoint_index}, "
                f"warmup_ticks={warmup_steps}"
            )

            return

        raise RuntimeError(
            f"Baseline failed to reach section start "
            f"after {max_baseline_retries} snapshot "
            f"respawn attempts."
        )

    def render_camera(self):
        if not self.enable_visualization:
            return
        if self.camera_sensor is None:
            return

        camera_data = self.camera_sensor.get_last_observation()
        if camera_data is not None:
            self.viewer.render(camera_data)

    def render(self):
        return None

    def close(self):
        if self.is_closed:
            return

        self.is_closed = True

        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None

        if self.vehicle is not None:
            try:
                self.vehicle.close()
            except Exception:
                pass
            self.vehicle = None

        if self.async_loop is not None:
            self.async_loop.call_soon_threadsafe(
                self.async_loop.stop
            )

        if self.async_thread is not None:
            self.async_thread.join()
