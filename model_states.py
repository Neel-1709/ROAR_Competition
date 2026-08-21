import asyncio
import statistics
from collections import Counter

import carla
import numpy as np
import roar_py_carla
from stable_baselines3 import SAC

from environment import CustomEnv
from submission import filter_waypoints


SECTION_NUMBER = 8
SECTION_START_INDEX = 1744
SECTION_END_INDEX = 2200
SPAWN_WAYPOINT_INDEX = SECTION_START_INDEX - 50

TICK_REPEAT = 3
TIME_TO_BEAT = 14.25
NUM_EPISODES = 30

MODEL_PATH = "models/section_8/sac_section_8_final"

CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000
CARLA_TIMEOUT = 10.0

CONTROL_STEP = 0.05
PHYSICS_STEP = 0.005

MAX_LATERAL_OFFSET_M = 0
MAX_YAW_OFFSET_DEG = 1.0
MIN_SPEED_SCALE = 0.97
MAX_SPEED_SCALE = 1.03

RANDOM_SEED = 42


def pct(x):
    return f"{100.0 * x:.1f}%"


async def apply_evaluation_perturbation(
    env,
    lateral_offset_m,
    yaw_offset_rad,
    speed_scale,
):
    location = np.asarray(
        env.vehicle.get_3d_location(),
        dtype=np.float64,
    ).copy()

    rpy = np.asarray(
        env.vehicle.get_roll_pitch_yaw(),
        dtype=np.float64,
    ).copy()

    velocity = np.asarray(
        env.velocity_sensor.get_last_gym_observation(),
        dtype=np.float64,
    ).copy()

    yaw = float(rpy[2])

    lateral_direction = np.array(
        [-np.sin(yaw), np.cos(yaw), 0.0],
        dtype=np.float64,
    )

    location += lateral_offset_m * lateral_direction
    rpy[2] += yaw_offset_rad
    velocity *= speed_scale

    env.vehicle.set_transform(location, rpy)
    env.vehicle.set_linear_3d_velocity(velocity)
    env.vehicle.set_angular_velocity(np.zeros(3))

    neutral_control = {
        "throttle": 0.0,
        "steer": 0.0,
        "brake": 0.0,
        "hand_brake": 0,
        "reverse": 0,
        "target_gear": 1,
    }

    await env.vehicle.apply_action(neutral_control)
    await env.world.step()
    await env.vehicle.receive_observation()

    vehicle_location = env.location_sensor.get_last_gym_observation()

    env.current_waypoint_index = filter_waypoints(
        vehicle_location,
        env.current_waypoint_index,
        env.maneuverable_waypoints,
    )

    env.previous_yaw = None
    env.old_progress = env.calculate_section_progress()


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    print("Connecting to CARLA...")

    carla_client = carla.Client(CARLA_HOST, CARLA_PORT)
    carla_client.set_timeout(CARLA_TIMEOUT)

    roar_instance = roar_py_carla.RoarPyCarlaInstance(carla_client)
    world = roar_instance.world

    world.set_control_steps(CONTROL_STEP, PHYSICS_STEP)
    world.set_asynchronous(False)

    env = CustomEnv(
        world=world,
        section_start_index=SECTION_START_INDEX,
        section_end_index=SECTION_END_INDEX,
        spawn_waypoint_index=SPAWN_WAYPOINT_INDEX,
        tick_repeat=TICK_REPEAT,
        time_to_beat=TIME_TO_BEAT,
        section_number=SECTION_NUMBER,
    )

    try:
        env.setup()

        print(f"Loading model: {MODEL_PATH}")
        model = SAC.load(MODEL_PATH, env=env)

        episode_rewards = []
        episode_lengths = []
        final_progresses = []
        successful_times = []

        crash_waypoints = Counter()
        failure_reasons = Counter()

        completions = 0
        crashes = 0

        print()
        print("=" * 76)
        print("ROBUSTNESS EVALUATION")
        print("=" * 76)
        print(f"Episodes:              {NUM_EPISODES}")
        print("Policy:                deterministic")
        print(f"Lateral perturbation:  ±{MAX_LATERAL_OFFSET_M:.2f} m")
        print(f"Yaw perturbation:      ±{MAX_YAW_OFFSET_DEG:.1f} deg")
        print(
            f"Speed perturbation:    "
            f"{MIN_SPEED_SCALE:.3f}x to {MAX_SPEED_SCALE:.3f}x"
        )
        print(f"Random seed:           {RANDOM_SEED}")
        print("=" * 76)
        print()

        for episode in range(1, NUM_EPISODES + 1):
            env.spawn_waypoint_index = np.random.randint(
                SECTION_START_INDEX - 100,
                SECTION_START_INDEX - 30
            )
            env.reset()

            lateral_offset = rng.uniform(
                -MAX_LATERAL_OFFSET_M,
                MAX_LATERAL_OFFSET_M,
            )

            yaw_offset_deg = rng.uniform(
                -MAX_YAW_OFFSET_DEG,
                MAX_YAW_OFFSET_DEG,
            )

            speed_scale = rng.uniform(
                MIN_SPEED_SCALE,
                MAX_SPEED_SCALE,
            )

            future = asyncio.run_coroutine_threadsafe(
                apply_evaluation_perturbation(
                    env,
                    lateral_offset,
                    np.radians(yaw_offset_deg),
                    speed_scale,
                ),
                env.async_loop,
            )
            future.result()

            obs = env.get_observation()
            env.section_start_time = env.world.last_tick_elapsed_seconds

            terminated = False
            truncated = False
            ep_reward = 0.0
            ep_len = 0
            last_info = {}

            while not (terminated or truncated):
                action, _ = model.predict(
                    obs,
                    deterministic=False,
                )

                obs, reward, terminated, truncated, info = env.step(action)

                ep_reward += float(reward)
                ep_len += 1
                last_info = info

            completed = bool(last_info.get("completed", False))
            crashed = bool(last_info.get("crashed", False))
            progress = float(
                last_info.get(
                    "progress",
                    env.calculate_section_progress(),
                )
            )
            section_time = float(last_info.get("time", 0.0))
            final_wp = int(env.current_waypoint_index)
            reason = last_info.get("reason", "Unknown")

            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_len)
            final_progresses.append(progress)

            if completed:
                completions += 1
                successful_times.append(section_time)
            else:
                failure_reasons[reason] += 1

            if crashed:
                crashes += 1
                crash_waypoints[final_wp] += 1

            print(
                f"Episode {episode:>3}/{NUM_EPISODES} | "
                f"{'COMPLETE' if completed else 'FAILED':<8} | "
                f"progress={pct(progress):>6} | "
                f"wp={final_wp:>4} | "
                f"time={section_time:>6.2f}s | "
                f"reward={ep_reward:>8.1f} | "
                f"lat={lateral_offset:+.3f}m | "
                f"yaw={yaw_offset_deg:+.2f}deg | "
                f"speed={speed_scale:.3f}x"
            )

        completion_rate = completions / NUM_EPISODES
        crash_rate = crashes / NUM_EPISODES

        print()
        print("=" * 76)
        print("EVALUATION SUMMARY")
        print("=" * 76)
        print(f"Episodes:                 {NUM_EPISODES}")
        print(
            f"Completions:              "
            f"{completions}/{NUM_EPISODES} "
            f"({pct(completion_rate)})"
        )
        print(
            f"Crashes:                  "
            f"{crashes}/{NUM_EPISODES} "
            f"({pct(crash_rate)})"
        )
        print(
            f"Mean final progress:      "
            f"{pct(statistics.mean(final_progresses))}"
        )
        print(
            f"Median final progress:    "
            f"{pct(statistics.median(final_progresses))}"
        )
        print(
            f"Mean episode reward:      "
            f"{statistics.mean(episode_rewards):.2f}"
        )
        print(
            f"Median episode reward:    "
            f"{statistics.median(episode_rewards):.2f}"
        )
        print(
            f"Mean episode length:      "
            f"{statistics.mean(episode_lengths):.1f} PPO steps"
        )

        if successful_times:
            print()
            print("SUCCESSFUL SECTION TIMES")
            print(
                f"  Mean:                   "
                f"{statistics.mean(successful_times):.3f}s"
            )
            print(
                f"  Median:                 "
                f"{statistics.median(successful_times):.3f}s"
            )
            print(
                f"  Best:                   "
                f"{min(successful_times):.3f}s"
            )
            print(
                f"  Worst:                  "
                f"{max(successful_times):.3f}s"
            )

            beats = sum(
                t <= TIME_TO_BEAT
                for t in successful_times
            )

            print(
                f"  Beat {TIME_TO_BEAT:.3f}s:          "
                f"{beats}/{len(successful_times)} "
                f"({pct(beats / len(successful_times))})"
            )
        else:
            print()
            print("No successful section completions.")

        print()
        print("MOST COMMON CRASH WAYPOINTS")

        if crash_waypoints:
            for waypoint, count in crash_waypoints.most_common(10):
                print(
                    f"  wp {waypoint}: "
                    f"{count}/{crashes} crashes"
                )
        else:
            print("  None")

        print()
        print("FAILURE REASONS")

        if failure_reasons:
            for reason, count in failure_reasons.most_common():
                print(f"  {reason}: {count}")
        else:
            print("  None")

        print()
        print("PROGRESS MILESTONES")

        for threshold in [0.25, 0.50, 0.75, 0.90, 0.95]:
            count = sum(
                p >= threshold
                for p in final_progresses
            )

            print(
                f"  Reached {pct(threshold):>6}: "
                f"{count}/{NUM_EPISODES} "
                f"({pct(count / NUM_EPISODES)})"
            )

        print("=" * 76)

    finally:
        print("Closing environment...")

        try:
            env.close()
        except Exception as exc:
            print(f"Warning while closing environment: {exc}")


if __name__ == "__main__":
    main()
