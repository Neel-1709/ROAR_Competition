
from collections import deque
import copy
from functools import reduce
import json
import os
from typing import List, Tuple, Dict, Optional
import math
import numpy as np
import roar_py_interface
from LateralController import LatController
from ThrottleController import ThrottleController
from WaypointLine import WaypointLine
from SectionStats import SectionStats
import atexit
from pathlib import Path
from stable_baselines3 import SAC

# from scipy.interpolate import interp1d

useDebug = False
useDebugPrinting = False
debugData = {}
dbg_carLocations = []
dbg_wpsToFollow = []
dbg_str = []
dbg_str2 = []
dbg_steer = []

# SAC deployment configuration. Baseline behavior is unchanged outside this range.
SAC_TAKEOVER_WP = 1745
SAC_PROGRESS_START_WP = 1744
SAC_END_WP = 2359
SAC_MAX_SAFE_STEER = 0.075
SAC_MAX_SAFE_HEADING_ERROR = 0.055
SAC_MAX_SAFE_LATERAL_ERROR = 0.55
SAC_SECTION_NUMBER = 8
SAC_ACTION_REPEAT = 3
SAC_MIN_LOOKAHEAD = 5.0
SAC_MAX_LOOKAHEAD = 40.0
SAC_MAX_OFFSET_METERS = 1.8
SAC_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "checkpoints"
    / "Final_Models"
    / "sac_section_8_410000_steps_copy.zip"
)


def dist_to_waypoint(location, waypoint: roar_py_interface.RoarPyWaypoint):
    return np.linalg.norm(location[:2] - waypoint.location[:2])


def filter_waypoints(
    location: np.ndarray,
    current_idx: int,
    waypoints: List[roar_py_interface.RoarPyWaypoint],
) -> int:
    for i in range(current_idx, len(waypoints) + current_idx):
        if dist_to_waypoint(location, waypoints[i % len(waypoints)]) < 3:
            return i % len(waypoints)
    min_dist = 1000
    min_ind = current_idx
    for i in range(0, 20):
        ind = (current_idx + i) % len(waypoints)
        d = dist_to_waypoint(location, waypoints[ind])
        if d < min_dist:
            min_dist = d
            min_ind = ind
    return min_ind


def findClosestIndex(location, waypoints: List[roar_py_interface.RoarPyWaypoint]):
    lowestDist = 100
    closestInd = 0
    for i in range(0, len(waypoints)):
        dist = dist_to_waypoint(location, waypoints[i % len(waypoints)])
        if dist < lowestDist:
            lowestDist = dist
            closestInd = i
    return closestInd % len(waypoints)


@atexit.register
def saveDebugData():
    print("Saving...")
    fname = "\\debugData\\line.txt"
    with open(
        f"{os.path.dirname(__file__)}{fname}", "w+"
    ) as outfile:
        outfile.write("\n--- Debug steer\n")
        for line in dbg_steer:
            outfile.write(f"{line}\n")
        outfile.write("\n--- Locatons\n")
        for line in dbg_carLocations:
            outfile.write(f"{line}\n")
        outfile.write("\n--- wpsToFollow\n")
        for line in dbg_wpsToFollow:
            outfile.write(f"{line}\n")
        outfile.write("\n--- Debug str\n")
        for line in dbg_str2:
            outfile.write(f"{line}\n")
        outfile.write("\n--- More Debug str\n")
        for line in dbg_str:
            outfile.write(f"{line}\n")
    print(f"Saved. {fname}")

    if useDebug:
        print("Saving debug data")
        jsonData = json.dumps(debugData, indent=4)
        with open(
            f"{os.path.dirname(__file__)}\\debugData\\debugData.json", "w+"
        ) as outfile:
            outfile.write(jsonData)
        print("Debug Data Saved")


class RoarCompetitionSolution:
    def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        vehicle: roar_py_interface.RoarPyActor,
        camera_sensor: roar_py_interface.RoarPyCameraSensor = None,
        location_sensor: roar_py_interface.RoarPyLocationInWorldSensor = None,
        velocity_sensor: roar_py_interface.RoarPyVelocimeterSensor = None,
        rpy_sensor: roar_py_interface.RoarPyRollPitchYawSensor = None,
        occupancy_map_sensor: roar_py_interface.RoarPyOccupancyMapSensor = None,
        collision_sensor: roar_py_interface.RoarPyCollisionSensor = None,
    ) -> None:
        self.maneuverable_waypoints = maneuverable_waypoints
        self.vehicle = vehicle
        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor
        self.lat_controller = LatController()
        self.throttle_controller = ThrottleController()
        self.section_stats = None
        self.section_indeces = []
        self.num_ticks = 0
        self.current_section = 0
        self.lapNum = 1
        self.previous_waypoint_to_follow = None
        self.max_radius = 10000
        self.previous_location = None
        self.total_dist = 0
        self.waypoint_line = WaypointLine()
        self.previous_brake = False
        self.s3_mult = 1

        # SAC-only state. None of this is consulted by the original baseline path.
        self.sac_model = None
        self.sac_active = False
        self.sac_previous_action = np.zeros(2, dtype=np.float32)
        self.sac_previous_yaw = None
        self.sac_cached_action = np.zeros(2, dtype=np.float32)
        self.sac_ticks_remaining = 0
        self.sac_path_yaw = None
        self.sac_lat_controller = None
        self.sac_throttle_controller = None
        self.sac_disabled_for_race = False
        self.last_route_waypoint_idx = None

    async def initialize(self) -> None:
        # NOTE waypoints are changed through this line
        self.maneuverable_waypoints = (
            roar_py_interface.RoarPyWaypoint.load_waypoint_list(
                np.load(f"{os.path.dirname(__file__)}\\waypoints\\waypointsPrimary.npz")
            )[35:]
        )
        self.section_stats = SectionStats(
            self.maneuverable_waypoints, self.location_sensor, self.velocity_sensor)

        sectionLocations = [
            [-278, 372], # Section 0 start location
            [64, 890], # Section 1 start location
            [511, 1037], # Section 2 start location
            [762, 908], # Section 3 start location
            [198, 307], # Section 4 start location
            [-11, 60], # Section 5 start location
            [-85, -339], # Section 6 start location
            [-210, -1060], # Section 7 start location 
            [-318, -991], # Section 8 start location
            [-352, -119], # Section 9 start location
        ]
        # for i in sectionLocations:
        #     self.section_indeces.append(
        #         findClosestIndex(i, self.maneuverable_waypoints)
        #     )
        self.section_indeces = [2611, 322, 557, 739, 1158, 1317, 1516, 1881, 1944, 2359]

        print(f"True total length: {len(self.maneuverable_waypoints) * 3}")
        print(f"1 lap length: {len(self.maneuverable_waypoints)}")
        print(f"Section indexes: {self.section_indeces}")
        print("\nLap 1\n")

        # Receive location, rotation and velocity data
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()

        self.current_waypoint_idx = 0
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location, self.current_waypoint_idx, self.maneuverable_waypoints
        )
        self.previous_location = vehicle_location

        # SAC inference setup. This does not alter any baseline controller/path state.
        self.sac_path_yaw = np.empty(len(self.maneuverable_waypoints), dtype=np.float64)
        for i, current_wp in enumerate(self.maneuverable_waypoints):
            next_wp = self.maneuverable_waypoints[(i + 1) % len(self.maneuverable_waypoints)]
            dx = next_wp.location[0] - current_wp.location[0]
            dy = next_wp.location[1] - current_wp.location[1]
            self.sac_path_yaw[i] = np.arctan2(dy, dx)

        if not SAC_MODEL_PATH.exists():
            raise FileNotFoundError(f"SAC model not found: {SAC_MODEL_PATH.resolve()}")
        self.sac_model = SAC.load(str(SAC_MODEL_PATH), device="auto")
        print(
            f"[SAC] Loaded {SAC_MODEL_PATH.resolve()} "
            f"timesteps={self.sac_model.num_timesteps}"
        )


    async def step(self) -> None:
        """Route to untouched baseline or SAC based on the current physical waypoint."""
        vehicle_location = self.location_sensor.get_last_gym_observation()
        candidate_idx = filter_waypoints(
            vehicle_location, self.current_waypoint_idx, self.maneuverable_waypoints
        )

        # New lap: clear stale SAC inference state. A safety lockout, if triggered,
        # intentionally remains active for the rest of the race.
        if (
            self.last_route_waypoint_idx is not None
            and self.last_route_waypoint_idx > 2400
            and candidate_idx < 200
        ):
            self.sac_active = False
            self.sac_previous_action = np.zeros(2, dtype=np.float32)
            self.sac_previous_yaw = None
            self.sac_cached_action = np.zeros(2, dtype=np.float32)
            self.sac_ticks_remaining = 0
            self.sac_lat_controller = None
            self.sac_throttle_controller = None

        self.last_route_waypoint_idx = candidate_idx

        should_use_sac = (
            not self.sac_disabled_for_race
            and SAC_TAKEOVER_WP <= candidate_idx < SAC_END_WP
        )

        if should_use_sac:
            self.current_waypoint_idx = candidate_idx
            if not self.sac_active:
                self._enter_sac_mode()
            return await self._sac_step()

        if self.sac_active:
            self.current_waypoint_idx = candidate_idx
            self._exit_sac_mode()

        # This is the original baseline step, unchanged.
        return await self._baseline_step_original()

    async def _baseline_step_original(self) -> None:
        """
        This function is called every world step.
        Note: You should not call receive_observation() on any sensor here, instead use get_last_observation() to get the last received observation.
        You can do whatever you want here, including apply_action() to the vehicle.
        """
        self.num_ticks += 1
        self.section_stats.step()

        # Receive location, rotation and velocity data
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        vehicle_velocity_norm = np.linalg.norm(vehicle_velocity)
        current_speed_kmh = vehicle_velocity_norm * 3.6

        # Find the waypoint closest to the vehicle
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location, self.current_waypoint_idx, self.maneuverable_waypoints
        )

        for i, section_ind in enumerate(self.section_indeces):
            if (
                abs(self.current_waypoint_idx - section_ind) <= 2
                and i != self.current_section
            ):
                self.current_section = i
                if self.current_section == 0 and self.lapNum != 3:
                    self.lapNum += 1

        nextWaypointIndex = self.get_lookahead_index(current_speed_kmh)
        waypoint_to_follow = self.next_waypoint_smooth(current_speed_kmh, vehicle_location)
        waypoint_to_follow_location = waypoint_to_follow.location
        snap_to_line_location = self.waypoint_line.get_next_waypoint_location(waypoint_to_follow.location)
        if self.current_section  not in [0, 9]:
            waypoint_to_follow_location = snap_to_line_location

        # Pure pursuit controller to steer the vehicle
        steer_control, steer_debug = self.lat_controller.run(
            vehicle_location, vehicle_rotation, waypoint_to_follow_location, self.current_waypoint_idx
        )

        # Custom controller to control the vehicle's speed
        waypoints_for_throttle = (self.maneuverable_waypoints * 2)[
            nextWaypointIndex : nextWaypointIndex + 300
        ]
        num_points_before_lookahead = 9
        wp_len = len(self.maneuverable_waypoints)
        wp_ind_for_throttle = ((nextWaypointIndex + wp_len) - num_points_before_lookahead) % wp_len
        additional_waypoints = (self.maneuverable_waypoints * 2)[
            wp_ind_for_throttle : wp_ind_for_throttle + 300
        ]
        throttle, brake, gear, speed_data, throttle_debug_str = self.throttle_controller.run(
            waypoints_for_throttle,
            vehicle_location,
            current_speed_kmh,
            self.current_section,
            additional_waypoints,
        )

        steerMultiplier = round((current_speed_kmh + 0.001) / 120, 3)
        
        if self.current_waypoint_idx in [800, 801]:
            self.s3_mult = 0.85
            if current_speed_kmh >= 162:
                self.s3_mult = 0.95
                if not self.previous_brake:
                    throttle = 0
                    brake = 1
                    self.previous_brake = True
            if current_speed_kmh < 160:
                self.s3_mult = 0.75
            print(f"spd {current_speed_kmh} mult{self.s3_mult} sec={self.current_section}")
        if self.current_waypoint_idx in [802, 803, 804]:
            self.previous_brake = False

        if self.current_section == 2:
            steerMultiplier *= 1.2
        if self.current_section in [3]:
            if self.current_waypoint_idx < 813:
                steerMultiplier *= self.s3_mult
            elif self.current_waypoint_idx < 845:
                steerMultiplier *= 1.42
            else:
                steerMultiplier *= 1
                self.s3_mult = 1

        if self.current_section == 4:
            steerMultiplier *= 1.65
            steerMultiplier = min(1.45, steerMultiplier)
        if self.current_section == 5:
            steerMultiplier *= 1.1
        if self.current_section in [6]:
            steerMultiplier = np.clip(steerMultiplier * 2.9, 2.8, 5.5)
        if self.current_section == 7:
            steerMultiplier *= 1.68

        if self.current_section == 9:
            if self.current_waypoint_idx > 2580:
                steerMultiplier = max(steerMultiplier, 1.62)
            else:
                steerMultiplier = max(steerMultiplier, 1.47)

        steer_value = np.clip(steer_control * steerMultiplier, -1, 1)
        # sec3
        if  820 < self.current_waypoint_idx < 837:
            steer_value = np.clip(steer_control * steerMultiplier, -0.007, 1)
        if self.current_waypoint_idx in [2381, 2382] and current_speed_kmh > 257:
            if not self.previous_brake:
              throttle = 0
              brake = 1
              self.previous_brake = True
        if self.current_waypoint_idx in [2383, 2384, 2385]:
            self.previous_brake = False

        control = {
            "throttle": np.clip(throttle, 0, 1),
            "steer": steer_value,
            "brake": np.clip(brake, 0, 1),
            "hand_brake": 0,
            "reverse": 0,
            "target_gear": gear,  # Gears do not appear to have an impact on speed
        }
        
        if useDebug:
            dbg_carLocations.append(f"{vehicle_location[0]}, {vehicle_location[1]}")
            dbg_wpsToFollow.append(f"{waypoint_to_follow_location[0]}, {waypoint_to_follow_location[1]}")

            self.total_dist += np.linalg.norm(vehicle_location - self.previous_location)
            self.previous_location = vehicle_location
            s = f"{self.total_dist:.0f}, {current_speed_kmh:.0f}, {speed_data.recommended_speed_now:.0f}, {speed_data.name}, {brake*10:.2f}"
            dbg_str.append(s)
            wp_ind = (self.lapNum-1)*3000 + self.current_waypoint_idx
            s = f"{wp_ind:.0f}, {current_speed_kmh:.0f}, {speed_data.recommended_speed_now:.0f}, {speed_data.name}, {brake*10:.2f}"
            dbg_steer.append(s)

            wpl = waypoint_to_follow_location
            d = np.linalg.norm(waypoint_to_follow.location - vehicle_location)
            s = f"d {self.total_dist:.0f} t {self.num_ticks} ind {self.current_waypoint_idx} \
sp {current_speed_kmh:.2f} rec {speed_data.recommended_speed_now:.1f} dif {(current_speed_kmh - speed_data.recommended_speed_now):.1f} \
r={speed_data.r:.0f}: {throttle_debug_str}, \
t {control['throttle']:.3f} \
br {control['brake']:.3f} \
st: {control['steer']:.10f}, \
{steer_control:.6f}, {steerMultiplier:.6f} trgt wp:ind {nextWaypointIndex} {nextWaypointIndex - self.current_waypoint_idx} {d:.1f} \
loc: ({vehicle_location[0]:.2f}, {vehicle_location[1]:.2f}) wp({wpl[0]:.1f}, {wpl[1]:.1f}) {steer_debug} section {self.current_section}"
            dbg_str2.append(s)


        if useDebug:
            debugData[self.num_ticks] = {}
            debugData[self.num_ticks]["loc"] = [
                round(vehicle_location[0].item(), 3),
                round(vehicle_location[1].item(), 3),
            ]
            debugData[self.num_ticks]["throttle"] = round(float(control["throttle"]), 3)
            debugData[self.num_ticks]["brake"] = round(float(control["brake"]), 3)
            debugData[self.num_ticks]["steer"] = round(float(control["steer"]), 10)
            debugData[self.num_ticks]["speed"] = round(current_speed_kmh, 3)
            debugData[self.num_ticks]["lap"] = self.lapNum

#             if useDebugPrinting and self.num_ticks % 5 == 0:
#                 print(
#                     f"- Target waypoint: ({waypoint_to_follow.location[0]:.2f}, {waypoint_to_follow.location[1]:.2f}) index {nextWaypointIndex} \n\
# Current location: ({vehicle_location[0]:.2f}, {vehicle_location[1]:.2f}) index {self.current_waypoint_idx} section {self.current_section} \n\
# Distance to target waypoint: {math.sqrt((waypoint_to_follow.location[0] - vehicle_location[0]) ** 2 + (waypoint_to_follow.location[1] - vehicle_location[1]) ** 2):.3f}\n"
#                 )

#                 print(
#                     f"--- Speed: {current_speed_kmh:.2f} kph \n\
# Throttle: {control['throttle']:.3f} \n\
# Brake: {control['brake']:.3f} \n\
# Steer: {control['steer']:.10f} \n"
#                 )

        await self.vehicle.apply_action(control)
        return control

    def _enter_sac_mode(self):
        # SAC gets its own fresh low-level controllers, matching CustomEnv, while
        # the original baseline controllers remain completely untouched.
        self.sac_lat_controller = LatController()
        self.sac_throttle_controller = ThrottleController()
        self.sac_previous_action = np.zeros(2, dtype=np.float32)
        self.sac_previous_yaw = None
        self.sac_cached_action = np.zeros(2, dtype=np.float32)
        self.sac_ticks_remaining = 0
        self.sac_active = True
        print(f"[SAC] TAKEOVER at waypoint {self.current_waypoint_idx}")

    def _exit_sac_mode(self):
        # Baseline state has been shadow-updated during SAC, so do not reset any
        # baseline controller or WaypointLine state here. Only clear SAC state.
        self.sac_previous_action = np.zeros(2, dtype=np.float32)
        self.sac_previous_yaw = None
        self.sac_cached_action = np.zeros(2, dtype=np.float32)
        self.sac_ticks_remaining = 0
        self.sac_lat_controller = None
        self.sac_throttle_controller = None
        self.sac_active = False
        print(f"[SAC] HANDOFF at waypoint {self.current_waypoint_idx}, baseline section={self.current_section}")

    def _shadow_baseline_update(self, vehicle_location, vehicle_rotation, current_speed_kmh):
        """Advance baseline internal state while SAC owns the actual control.

        This intentionally mirrors the stateful parts of the original baseline step,
        but does NOT apply the baseline control to the vehicle.
        """
        # Keep section/lap bookkeeping current across boundaries skipped by SAC.
        for i, section_ind in enumerate(self.section_indeces):
            if (
                abs(self.current_waypoint_idx - section_ind) <= 2
                and i != self.current_section
            ):
                self.current_section = i
                if self.current_section == 0 and self.lapNum != 3:
                    self.lapNum += 1

        # Reproduce baseline target generation so WaypointLine advances exactly as
        # it would in a normal baseline tick.
        nextWaypointIndex = self.get_lookahead_index(current_speed_kmh)
        waypoint_to_follow = self.next_waypoint_smooth(current_speed_kmh, vehicle_location)
        waypoint_to_follow_location = waypoint_to_follow.location
        snap_to_line_location = self.waypoint_line.get_next_waypoint_location(
            waypoint_to_follow.location
        )
        if self.current_section not in [0, 9]:
            waypoint_to_follow_location = snap_to_line_location

        # Advance any internal lateral-controller history. Discard output.
        self.lat_controller.run(
            vehicle_location,
            vehicle_rotation,
            waypoint_to_follow_location,
            self.current_waypoint_idx,
        )

        # Advance throttle-controller history using the original baseline horizon.
        waypoints_for_throttle = (self.maneuverable_waypoints * 2)[
            nextWaypointIndex : nextWaypointIndex + 300
        ]
        wp_len = len(self.maneuverable_waypoints)
        wp_ind_for_throttle = (
            (nextWaypointIndex + wp_len) - 9
        ) % wp_len
        additional_waypoints = (self.maneuverable_waypoints * 2)[
            wp_ind_for_throttle : wp_ind_for_throttle + 300
        ]
        self.throttle_controller.run(
            waypoints_for_throttle,
            vehicle_location,
            current_speed_kmh,
            self.current_section,
            additional_waypoints,
        )

    async def _sac_step(self):
        # This is CustomEnv.step(action) split across competition-runner world ticks.
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        current_speed_kmh = np.linalg.norm(vehicle_velocity) * 3.6

        self.current_waypoint_idx = filter_waypoints(
            vehicle_location, self.current_waypoint_idx, self.maneuverable_waypoints
        )

        action = self._get_or_update_sac_action()
        lookahead_distance = (
            (action[0] + 1.0) / 2.0
            * (SAC_MAX_LOOKAHEAD - SAC_MIN_LOOKAHEAD)
            + SAC_MIN_LOOKAHEAD
        )
        lateral_offset = action[1] * SAC_MAX_OFFSET_METERS

        waypoints_to_shift = self._sac_waypoints_until_end()
        if not waypoints_to_shift:
            # The runner can skip directly to the handoff boundary between ticks.
            self._exit_sac_mode()
            return await self._baseline_step_original()

        future_waypoints = self._sac_shift_waypoint_path(
            lateral_offset, lookahead_distance, waypoints_to_shift
        )
        target_local_idx = min(
            max(int(lookahead_distance) - 1, 0), len(future_waypoints) - 1
        )
        modified_wp = future_waypoints[target_local_idx]

        steer_control, _ = self.sac_lat_controller.run(
            vehicle_location,
            vehicle_rotation,
            modified_wp.location,
            self.current_waypoint_idx,
        )

        wp_len = len(self.maneuverable_waypoints)
        throttle_horizon = 300
        throttle_waypoints = [
            self.maneuverable_waypoints[(self.current_waypoint_idx + i) % wp_len]
            for i in range(1, throttle_horizon + 1)
        ]
        additional_start = (self.current_waypoint_idx - 9) % wp_len
        additional_waypoints = [
            self.maneuverable_waypoints[(additional_start + i) % wp_len]
            for i in range(throttle_horizon)
        ]

        throttle, brake, gear, _, _ = self.sac_throttle_controller.run(
            throttle_waypoints,
            vehicle_location,
            current_speed_kmh,
            SAC_SECTION_NUMBER,
            additional_waypoints,
        )

        steer_multiplier = round((current_speed_kmh + 0.001) / 120, 3)
        steer_value = float(np.clip(steer_control * steer_multiplier, -1, 1))

        # Safety shield: do not blend or modify SAC. If its raw training-time
        # control is already outside the envelope that succeeded in evaluation,
        # hand the remainder of the race back to the known-good baseline.
        current_yaw = float(vehicle_rotation[2])
        path_yaw = float(self.sac_path_yaw[self.current_waypoint_idx])
        heading_error = self._sac_normalize_angle(path_yaw - current_yaw)
        lateral_error = float(self._sac_get_lateral_error())

        unsafe_sac = (
            not np.isfinite(steer_value)
            or abs(steer_value) > SAC_MAX_SAFE_STEER
            or abs(heading_error) > SAC_MAX_SAFE_HEADING_ERROR
            or abs(lateral_error) > SAC_MAX_SAFE_LATERAL_ERROR
        )

        if unsafe_sac:
            print(
                f"[SAC] SAFETY FALLBACK at wp={self.current_waypoint_idx} "
                f"steer={steer_value:.4f} heading={heading_error:.4f} "
                f"lat={lateral_error:.3f}"
            )
            self.sac_disabled_for_race = True
            self._exit_sac_mode()
            # _exit_sac_mode clears SAC state only. The original baseline then
            # computes and applies this tick's control exactly as before.
            return await self._baseline_step_original()

        # This tick will actually be controlled by SAC. Count it once and keep
        # baseline state synchronized in shadow mode.
        self.num_ticks += 1
        self.section_stats.step()

        # Only shadow-update baseline state on ticks where SAC will actually own
        # the vehicle. This keeps handoff state current without double-updating
        # a baseline fallback tick.
        self._shadow_baseline_update(
            vehicle_location, vehicle_rotation, current_speed_kmh
        )

        control = {
            "throttle": np.clip(throttle, 0, 1),
            "steer": steer_value,
            "brake": np.clip(brake, 0, 1),
            "hand_brake": 0,
            "reverse": 0,
            "target_gear": gear,
        }
        await self.vehicle.apply_action(control)
        return control

    def _get_or_update_sac_action(self):
        if self.sac_ticks_remaining <= 0:
            observation = self._get_sac_observation()
            action, _ = self.sac_model.predict(observation, deterministic=True)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            if action.shape != (2,):
                raise RuntimeError(f"Expected SAC action shape (2,), got {action.shape}")
            action = np.clip(action, -1.0, 1.0)
            self.sac_cached_action = action.copy()
            # Training's next observation contains the action that just controlled
            # the preceding three world ticks.
            self.sac_previous_action = action.copy()
            self.sac_ticks_remaining = SAC_ACTION_REPEAT

        action = self.sac_cached_action.copy()
        self.sac_ticks_remaining -= 1
        return action

    def _get_sac_observation(self):
        # Mathematical copy of CustomEnv.get_observation().
        observation = np.empty((87,), dtype=np.float32)
        vehicle_rpy = self.rpy_sensor.get_last_gym_observation()
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()

        current_yaw = vehicle_rpy[2]
        if self.sac_previous_yaw is None:
            yaw_rate = 0.0
        else:
            yaw_change = self._sac_normalize_angle(current_yaw - self.sac_previous_yaw)
            yaw_rate = yaw_change / 0.05
        self.sac_previous_yaw = current_yaw
        observation[0] = np.clip(yaw_rate / 65.0, -1.0, 1.0)

        path_yaw = self.sac_path_yaw[self.current_waypoint_idx]
        heading_error = self._sac_normalize_angle(path_yaw - current_yaw)
        observation[1] = np.clip(heading_error / np.pi, -1.0, 1.0)

        speed_kmh = np.linalg.norm(vehicle_velocity) * 3.6
        observation[2] = np.clip(speed_kmh / 60.0, -1.0, 1.0)

        n = len(self.maneuverable_waypoints)
        previous_wp = self.maneuverable_waypoints[(self.current_waypoint_idx - 1) % n].location
        next_wp = self.maneuverable_waypoints[(self.current_waypoint_idx + 1) % n].location
        path = next_wp - previous_wp
        path_norm = np.sqrt(path[0] ** 2 + path[1] ** 2)
        if path_norm < 1e-8:
            lateral_error = 0.0
        else:
            lateral_error = (
                path[1] * vehicle_location[0]
                - path[0] * vehicle_location[1]
                + next_wp[0] * previous_wp[1]
                - next_wp[1] * previous_wp[0]
            ) / path_norm
        observation[3] = np.clip(lateral_error / 3.0, -1.0, 1.0)

        x_offsets = []
        y_offsets = []
        for i in range(1, 41):
            idx = (self.current_waypoint_idx + i * 5) % n
            waypoint = self.maneuverable_waypoints[idx]
            dx = waypoint.location[0] - vehicle_location[0]
            dy = waypoint.location[1] - vehicle_location[1]
            distance = np.sqrt(dx ** 2 + dy ** 2)
            global_angle = np.arctan2(dy, dx)
            local_angle = self._sac_normalize_angle(global_angle - current_yaw)
            x_offsets.append(np.clip(distance * np.cos(local_angle) / 400.0, -1.0, 1.0))
            y_offsets.append(np.clip(distance * np.sin(local_angle) / 300.0, -1.0, 1.0))

        observation[4:44] = x_offsets
        observation[44:84] = y_offsets
        observation[84] = self._sac_section_progress()
        observation[85:87] = self.sac_previous_action
        return observation

    def _sac_normalize_angle(self, angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def _sac_forward_distance(self, start_idx, end_idx):
        return (end_idx - start_idx) % len(self.maneuverable_waypoints)

    def _sac_section_progress(self):
        total = self._sac_forward_distance(SAC_PROGRESS_START_WP, SAC_END_WP)
        if total == 0:
            return 1.0
        progressed = self._sac_forward_distance(SAC_PROGRESS_START_WP, self.current_waypoint_idx)
        return float(np.clip(progressed / total, 0.0, 1.0))

    def _sac_waypoints_until_end(self):
        n = len(self.maneuverable_waypoints)
        remaining = self._sac_forward_distance(self.current_waypoint_idx, SAC_END_WP)
        if remaining == 0:
            return []
        return copy.deepcopy([
            self.maneuverable_waypoints[(self.current_waypoint_idx + i) % n]
            for i in range(1, remaining + 1)
        ])

    def _sac_get_lateral_error(self):
        vehicle_location = self.location_sensor.get_last_gym_observation()
        n = len(self.maneuverable_waypoints)
        previous_wp = self.maneuverable_waypoints[(self.current_waypoint_idx - 1) % n].location
        next_wp = self.maneuverable_waypoints[(self.current_waypoint_idx + 1) % n].location
        path = next_wp - previous_wp
        path_norm = np.sqrt(path[0] ** 2 + path[1] ** 2)
        if path_norm < 1e-8:
            return 0.0
        return (
            path[1] * vehicle_location[0]
            - path[0] * vehicle_location[1]
            + next_wp[0] * previous_wp[1]
            - next_wp[1] * previous_wp[0]
        ) / path_norm

    def _sac_shift_waypoint_path(self, shift_amount, lookahead_distance, waypoints_to_shift):
        # Exact copy of the training environment's shift_waypoint_path().
        shifted_waypoints = copy.deepcopy(waypoints_to_shift)
        before_len = min(int(lookahead_distance), len(shifted_waypoints))
        current_offset = self._sac_get_lateral_error()

        for i, wp in enumerate(shifted_waypoints):
            track_idx = (self.current_waypoint_idx + 1 + i) % len(self.maneuverable_waypoints)
            prev_wp = self.maneuverable_waypoints[(track_idx - 1) % len(self.maneuverable_waypoints)]
            next_wp = self.maneuverable_waypoints[(track_idx + 1) % len(self.maneuverable_waypoints)]
            path = next_wp.location - prev_wp.location
            angle = np.arctan2(path[1], path[0])
            perpendicular_angle = self._sac_normalize_angle(angle) + np.pi / 2.0

            if i < before_len:
                fraction = i / max(before_len - 1, 1)
                smoothstep = 3 * fraction ** 2 - 2 * fraction ** 3
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

    def get_lookahead_value(self, speed):
        """
        Returns the number of waypoints to look ahead based on the speed the car is currently going
        """
        speed_to_lookahead_dict = {
            90: 9,
            110: 11,
            130: 14,
            160: 18,
            180: 22,
            200: 26,
            250: 30,
            300: 35,
        }

        # Interpolation method
        # NOTE does not work as well as the dictionary lookahead method, likely to cause crashes.

        # speedBoundList = [0, 90, 110, 130, 160, 180, 200, 250, 300]
        # lookaheadList = [5, 11, 13, 15, 18, 22, 25, 28, 32]

        # interpolationFunction = interp1d(speedBoundList, lookaheadList)
        # return int(interpolationFunction(speed))

        for speed_upper_bound, num_points in speed_to_lookahead_dict.items():
            if speed < speed_upper_bound:
                return num_points
        return 8

    def get_lookahead_index(self, speed):
        """
        Adds the lookahead waypoint to the current waypoint and normalizes it so that the value does not go out of bounds
        """
        num_waypoints = self.get_lookahead_value(speed)
        # print("speed " + str(speed)
        #       + " cur_ind " + str(self.current_waypoint_idx)
        #       + " num_points " + str(num_waypoints)
        #       + " index " + str((self.current_waypoint_idx + num_waypoints) % len(self.maneuverable_waypoints)) )
        return (self.current_waypoint_idx + num_waypoints) % len(
            self.maneuverable_waypoints
        )

    # def get_lateral_pid_config(self):
    #     """
    #     Returns the PID values for the lateral (steering) PID
    #     """
    #     with open(
    #         f"{os.path.dirname(__file__)}\\configs\\LatPIDConfig.json", "r"
    #     ) as file:
    #         config = json.load(file)
    #     return config

    # The idea and code for averaging points is from smooth_waypoint_following_local_planner.py (Summer 2023)
    def next_waypoint_smooth(self, current_speed: float, vehicle_location: float):
        """
        If the speed is higher than 70, 'smooth out' the path that the car will take
        """
        if self.current_section == 3:
            kdd = 0.25
            distance = kdd * current_speed
            distance = np.clip(distance, 44, 70)
            location, _ = self.waypoint_line.get_lookahead_location(vehicle_location, distance)
            point = roar_py_interface.RoarPyWaypoint(location, roll_pitch_yaw=np.ndarray([0, 0, 0]), lane_width=0.0)
            return point
        if self.current_section in [5, 7]:
            kdd = 0.25
            distance = kdd * current_speed
            distance = np.clip(distance, 30, 70)
            location, _ = self.waypoint_line.get_lookahead_location(vehicle_location, distance)
            point = roar_py_interface.RoarPyWaypoint(location, roll_pitch_yaw=np.ndarray([0, 0, 0]), lane_width=0.0)
            return point
        if self.current_section in [6]:
            kdd = 0.29
            distance = kdd * current_speed
            distance = np.clip(distance, 32, 70)
            location, _ = self.waypoint_line.get_lookahead_location(vehicle_location, distance)
            point = roar_py_interface.RoarPyWaypoint(location, roll_pitch_yaw=np.ndarray([0, 0, 0]), lane_width=0.0)
            return point
        if current_speed > 70 and current_speed < 300:
            target_waypoint = self.average_point(current_speed)
        else:
            new_waypoint_index = self.get_lookahead_index(current_speed)
            target_waypoint = self.maneuverable_waypoints[new_waypoint_index]

        return target_waypoint

    def new_RoarPyWaypoint(self, location):
        return roar_py_interface.RoarPyWaypoint(location, roll_pitch_yaw=np.ndarray([0, 0, 0]), lane_width=12.0)


    def average_point(self, current_speed):
        """
        Returns a new averaged waypoint based on the location of a number of other waypoints
        """
        next_waypoint_index = self.get_lookahead_index(current_speed)
        lookahead_value = self.get_lookahead_value(current_speed)
        num_points = lookahead_value * 2

        # Section specific tuning
        if self.current_section == 0:
            num_points = round(lookahead_value * 1.5)
        if self.current_section == 3:
            next_waypoint_index = self.current_waypoint_idx + 22
            num_points = 35
        if self.current_section == 4:
            num_points = lookahead_value + 5
            next_waypoint_index = self.current_waypoint_idx + 24
        if self.current_section == 5:
            # num_points = round(lookahead_value * 1.1)
            num_points = lookahead_value
        if self.current_section == 6:
            num_points = lookahead_value
            # num_points = 5
            next_waypoint_index = self.current_waypoint_idx + 28
        if self.current_section == 7:
            # Jolt between sections 6 and 7 likely due to the differences in lookahead values and steering multipliers. 
            num_points = round(lookahead_value * 1.25)
        if self.current_section == 9:
            # (self.current_waypoint_idx + 8) % len(self.maneuverable_waypoints)
            num_points = 0

        start_index_for_avg = (next_waypoint_index - (num_points // 2)) % len(
            self.maneuverable_waypoints
        )

        next_waypoint_index = next_waypoint_index % len(self.maneuverable_waypoints)
        next_waypoint = self.maneuverable_waypoints[next_waypoint_index]
        next_location = next_waypoint.location

        sample_points = [
            (start_index_for_avg + i) % len(self.maneuverable_waypoints)
            for i in range(0, num_points)
        ]
        if num_points > 3:
            location_sum = reduce(
                lambda x, y: x + y,
                (self.maneuverable_waypoints[i].location for i in sample_points),
            )
            num_points = len(sample_points)
            new_location = location_sum / num_points
            shift_distance = np.linalg.norm(next_location - new_location)
            max_shift_distance = 2.0
            if self.current_section == 1:
                max_shift_distance = 0.2
            if shift_distance > max_shift_distance:
                uv = (new_location - next_location) / shift_distance
                new_location = next_location + uv * max_shift_distance

            target_waypoint = roar_py_interface.RoarPyWaypoint(
                location=new_location,
                roll_pitch_yaw=np.ndarray([0, 0, 0]),
                lane_width=0.0,
            )
            # if next_waypoint_index > 1900 and next_waypoint_index < 2300:
            #   print("AVG: next_ind:" + str(next_waypoint_index) + " next_loc: " + str(next_location)
            #       + " new_loc: " + str(new_location) + " shift:" + str(shift_distance)
            #       + " num_points: " + str(num_points) + " start_ind:" + str(start_index_for_avg)
            #       + " curr_speed: " + str(current_speed))

        else:
            target_waypoint = self.maneuverable_waypoints[next_waypoint_index]

        return target_waypoint