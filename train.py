import carla
import roar_py_carla

from stable_baselines3.common.env_checker import check_env

import numpy as np

from environment import CustomEnv
from model import get_model, save_model, get_checkpoint_callback


SECTION_NUMBER = 8

SECTION_START_INDEX = 1744
SECTION_END_INDEX = 2359
SPAWN_WAYPOINT_INDEX = 0

TICK_REPEAT = 3
TIME_TO_BEAT = 14.25

TOTAL_TIMESTEPS = 1_000
LOAD_EXISTING_MODEL = False
CHECKPOINT_FREQUENCY = 5_000

CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000
CARLA_TIMEOUT = 10.0

CONTROL_STEP = 0.05
PHYSICS_STEP = 0.005


def main():
    print("Connecting to CARLA...")

    carla_client = carla.Client(CARLA_HOST, CARLA_PORT)
    carla_client.set_timeout(CARLA_TIMEOUT)

    roar_instance = roar_py_carla.RoarPyCarlaInstance(carla_client)
    world = roar_instance.world

    world.set_control_steps(CONTROL_STEP, PHYSICS_STEP)
    world.set_asynchronous(False)

    print("Creating environment...")

    env = CustomEnv(
        world=world,
        section_start_index=SECTION_START_INDEX,
        section_end_index=SECTION_END_INDEX,
        spawn_waypoint_index=SPAWN_WAYPOINT_INDEX,
        tick_repeat=TICK_REPEAT,
        time_to_beat=TIME_TO_BEAT,
        section_number=SECTION_NUMBER
    )

    try:
        print("Setting up environment...")
        env.setup()

        obs, info = env.reset()

        for _ in range(100):
            action = np.array([0.0, 0.0], dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                obs, info = env.reset()
        print("Environment passed check_env().")

        model = get_model(
            env=env,
            section_number=SECTION_NUMBER,
            load_existing=LOAD_EXISTING_MODEL,
        )

        checkpoint_callback = get_checkpoint_callback(
            section_number=SECTION_NUMBER,
            save_freq=CHECKPOINT_FREQUENCY,
        )

        print()
        print("========================================")
        print(f"Training PPO for section {SECTION_NUMBER}")
        print(f"Section: {SECTION_START_INDEX} -> {SECTION_END_INDEX}")
        print(f"Warmup spawn index: {SPAWN_WAYPOINT_INDEX}")
        print(f"Action repeat: {TICK_REPEAT} CARLA ticks")
        print(f"Time to beat: {TIME_TO_BEAT:.3f} s")
        print(f"Training timesteps: {TOTAL_TIMESTEPS}")
        print(
            "Mode:",
            "continue existing model"
            if LOAD_EXISTING_MODEL
            else "new model",
        )
        print("========================================")
        print()

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            reset_num_timesteps=not LOAD_EXISTING_MODEL,
            progress_bar=False,
        )

        save_model(
            model=model,
            section_number=SECTION_NUMBER,
        )

        print(f"Training complete for section {SECTION_NUMBER}.")

    finally:
        print("Closing environment...")

        try:
            env.close()
        except Exception as exc:
            print(f"Warning while closing environment: {exc}")


if __name__ == "__main__":
    main()
