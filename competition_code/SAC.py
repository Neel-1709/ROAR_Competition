import traceback
from pathlib import Path

import carla
import roar_py_carla

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from environment import CustomEnv


SECTION_NUMBER = 8

SECTION_START_INDEX = 1744
SECTION_END_INDEX = 2359
SPAWN_WAYPOINT_INDEX = SECTION_START_INDEX - 50

TICK_REPEAT = 3
TIME_TO_BEAT = 22.2

TOTAL_TIMESTEPS = 420_000
LOAD_EXISTING_MODEL = True
CHECKPOINT_FREQUENCY = 10_000

CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000
CARLA_TIMEOUT = 10.0

CONTROL_STEP = 0.05
PHYSICS_STEP = 0.005

LEARNING_RATE = 3e-4
BUFFER_SIZE = 100_000
LEARNING_STARTS = 2_000
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
TRAIN_FREQ = 1
GRADIENT_STEPS = 1
ENT_COEF = "auto"

POLICY_KWARGS = {
    "net_arch": [256, 256],
}

CHECKPOINT_DIR = Path("checkpoints") / f"section_{SECTION_NUMBER}" / "sac"
FINAL_MODEL_DIR = Path("models") / f"section_{SECTION_NUMBER}"
FINAL_MODEL_PATH = FINAL_MODEL_DIR / f"sac_section_{SECTION_NUMBER}_final"
LOG_DIR = Path("logs") / f"section_{SECTION_NUMBER}" / "SAC"

CHECKPOINT_TO_LOAD = Path("checkpoints/Final_Models/sac_section_8_410000_steps_copy.zip")


def get_latest_checkpoint():
    checkpoint_files = list(CHECKPOINT_DIR.glob("*.zip"))
    if not checkpoint_files:
        return None
    return max(checkpoint_files, key=lambda p: p.stat().st_mtime)


def create_new_model(env):
    print("Creating new SAC model...")

    return SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,
        tau=TAU,
        gamma=GAMMA,
        train_freq=TRAIN_FREQ,
        gradient_steps=GRADIENT_STEPS,
        ent_coef=ENT_COEF,
        policy_kwargs=POLICY_KWARGS,
        verbose=1,
        tensorboard_log=str(LOG_DIR),
        device="auto",
    )


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

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
        section_number=SECTION_NUMBER,
    )

    model = None


    try:
        print("Setting up environment...")
        env.setup()

        if LOAD_EXISTING_MODEL:
            if not CHECKPOINT_TO_LOAD.exists():
                raise FileNotFoundError(
                    f"Checkpoint not found: {CHECKPOINT_TO_LOAD}"
                )

            print(f"Loading SAC checkpoint: {CHECKPOINT_TO_LOAD}")

            model = SAC.load(
                str(CHECKPOINT_TO_LOAD),
                env=env,
                device="auto",
            )

            loaded_timesteps = int(model.num_timesteps)

            print(f"Loaded model at {loaded_timesteps} timesteps")

        else:
            model = create_new_model(env)
            loaded_timesteps = 0

        remaining_timesteps = max(
            TOTAL_TIMESTEPS - loaded_timesteps,
            0,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=CHECKPOINT_FREQUENCY,
            save_path=str(CHECKPOINT_DIR),
            name_prefix=f"sac_section_{SECTION_NUMBER}",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )

        print()
        print("========================================")
        print(f"Training SAC for section {SECTION_NUMBER}")
        print(f"Section: {SECTION_START_INDEX} -> {SECTION_END_INDEX}")
        print(f"Warmup spawn index: {SPAWN_WAYPOINT_INDEX}")
        print(f"Action repeat: {TICK_REPEAT} CARLA ticks")
        print(f"Time to beat: {TIME_TO_BEAT:.3f} s")
        print(f"Loaded timesteps: {loaded_timesteps}")
        print(f"Target cumulative timesteps: {TOTAL_TIMESTEPS}")
        print(f"Timesteps remaining: {remaining_timesteps}")
        print(
            "Mode:",
            "continue SAC checkpoint"
            if loaded_timesteps > 0
            else "new SAC model",
        )
        print("========================================")
        print()

        if remaining_timesteps <= 0:
            print("Model has already reached or exceeded the requested timestep target.")
            return

        model.learn(
            total_timesteps=remaining_timesteps,
            callback=checkpoint_callback,
            reset_num_timesteps=(loaded_timesteps == 0),
            progress_bar=False,
        )

        model.save(str(FINAL_MODEL_PATH))

        try:
            model.save_replay_buffer(
                str(FINAL_MODEL_PATH) + "_replay_buffer.pkl"
            )
        except Exception as exc:
            print(f"Warning: could not save final replay buffer: {exc}")

        print()
        print(f"Training complete. Final timestep count: {model.num_timesteps}")
        print(f"Final model saved to: {FINAL_MODEL_PATH}.zip")

    except KeyboardInterrupt:
        print()
        print("Training interrupted.")

        if model is not None:
            interrupted_path = (
                FINAL_MODEL_DIR
                / f"sac_section_{SECTION_NUMBER}_interrupted_{model.num_timesteps}"
            )

            print(f"Saving interrupted model to {interrupted_path}.zip")
            model.save(str(interrupted_path))

            try:
                replay_path = str(interrupted_path) + "_replay_buffer.pkl"
                model.save_replay_buffer(replay_path)
                print(f"Replay buffer saved to {replay_path}")
            except Exception as exc:
                print(f"Warning: could not save replay buffer: {exc}")

    except Exception as exc:
        print()
        print("TRAINING ERROR:")
        print(repr(exc))
        traceback.print_exc()
        raise

    finally:
        print("Closing environment...")

        try:
            env.close()
        except Exception as exc:
            print(f"Warning while closing environment: {exc}")


if __name__ == "__main__":
    main()

