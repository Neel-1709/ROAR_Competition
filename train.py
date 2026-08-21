import carla
import roar_py_carla

from stable_baselines3 import PPO

from environment import CustomEnv
from model import save_model, get_checkpoint_callback

from model import get_model

from pathlib import Path

SECTION_NUMBER = 8

# Section 8
SECTION_START_INDEX = 1744
SECTION_END_INDEX = 2050

# Spawn snapshot shortly before PPO takeover
SPAWN_WAYPOINT_INDEX = SECTION_START_INDEX - 50

TICK_REPEAT = 3
TIME_TO_BEAT = 5.8

# This is the FINAL cumulative timestep target
TOTAL_TIMESTEPS = 150_000

CHECKPOINT_PATH = "checkpoints/section_8/ppo_section_8_2000_steps"
CHECKPOINT_FREQUENCY = 1_000
CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000
CARLA_TIMEOUT = 10.0

CONTROL_STEP = 0.05
PHYSICS_STEP = 0.005


def main():
    print("Connecting to CARLA...")

    carla_client = carla.Client(
        CARLA_HOST,
        CARLA_PORT
    )
    carla_client.set_timeout(CARLA_TIMEOUT)

    roar_instance = roar_py_carla.RoarPyCarlaInstance(
        carla_client
    )

    world = roar_instance.world

    world.set_control_steps(
        CONTROL_STEP,
        PHYSICS_STEP
    )
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
        # --------------------------------------------------------
        # 1. Set up CARLA environment
        # --------------------------------------------------------
        print("Setting up environment...")
        env.setup()

        # DO NOT call env.reset() here.
        # Stable-Baselines3 will call reset() when learn() begins.

        # --------------------------------------------------------
        # 2. Load checkpoint
        # --------------------------------------------------------

        

        checkpoint_dir = Path("checkpoints") / f"section_{SECTION_NUMBER}"

        checkpoint_files = sorted(
            checkpoint_dir.glob("*.zip"),
            key=lambda p: p.stat().st_mtime,
        )

        if not checkpoint_files:
            raise FileNotFoundError(
                f"No checkpoint files found in {checkpoint_dir}"
            )

        checkpoint_path = checkpoint_files[-1]

        print(f"Loading latest checkpoint: {checkpoint_path}")

        model = PPO.load(
            str(checkpoint_path),
            env=env,
        )

        # model = get_model(
        #     env=env,
        #     section_number=SECTION_NUMBER,
        #     load_existing=False,
        # )

        loaded_timesteps = int(
            model.num_timesteps
        )

        remaining_timesteps = max(
            TOTAL_TIMESTEPS - loaded_timesteps,
            0,
        )

        print()
        print("========================================")
        print(f"Training PPO for section {SECTION_NUMBER}")
        print(
            f"Section: "
            f"{SECTION_START_INDEX} -> {SECTION_END_INDEX}"
        )
        print(
            f"Warmup spawn index: "
            f"{SPAWN_WAYPOINT_INDEX}"
        )
        print(
            f"Action repeat: "
            f"{TICK_REPEAT} CARLA ticks"
        )
        print(
            f"Time to beat: "
            f"{TIME_TO_BEAT:.3f} s"
        )
        print(
            f"Checkpoint timesteps: "
            f"{loaded_timesteps}"
        )
        print(
            f"Target cumulative timesteps: "
            f"{TOTAL_TIMESTEPS}"
        )
        print(
            f"Timesteps remaining: "
            f"{remaining_timesteps}"
        )
        print("Mode: continue checkpoint")
        print("========================================")
        print()

        # --------------------------------------------------------
        # 3. Make sure there is actually training left
        # --------------------------------------------------------
        if remaining_timesteps <= 0:
            print(
                "Checkpoint has already reached or exceeded "
                f"{TOTAL_TIMESTEPS} timesteps."
            )
            return

        # --------------------------------------------------------
        # 4. Checkpoint callback
        # --------------------------------------------------------
        checkpoint_callback = (
            get_checkpoint_callback(
                section_number=SECTION_NUMBER,
                save_freq=CHECKPOINT_FREQUENCY,
            )
        )

        # --------------------------------------------------------
        # 5. Continue training
        # --------------------------------------------------------
        model.learn(
            # With reset_num_timesteps=False, this is the
            # number of ADDITIONAL timesteps to train.
            total_timesteps=remaining_timesteps,
            callback=checkpoint_callback,
            reset_num_timesteps=False,
            progress_bar=False,
        )

        # --------------------------------------------------------
        # 6. Save final model
        # --------------------------------------------------------
        save_model(
            model=model,
            section_number=SECTION_NUMBER,
        )

        print()
        print(
            f"Training complete. "
            f"Final model timestep count: "
            f"{model.num_timesteps}"
        )

    except Exception as exc:
        print()
        print("TRAINING ERROR:")
        print(repr(exc))

        import traceback
        traceback.print_exc()

        raise

    except KeyboardInterrupt:
        print("\nTraining interrupted.")

        if model is not None:
            print("Saving interrupted model...")
            save_model(
                model=model,
                section_number=SECTION_NUMBER,
            )

    except KeyboardInterrupt:
        print("\nTraining interrupted.")

        # Save current state even on Ctrl+C
        if model is not None:
            print("Saving interrupted model...")

            save_model(
                model=model,
                section_number=SECTION_NUMBER,
            )

    finally:
        print("Closing environment...")

        try:
            env.close()
        except Exception as exc:
            print(
                f"Warning while closing environment: "
                f"{exc}"
            )


if __name__ == "__main__":
    main()