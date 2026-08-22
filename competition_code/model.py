from random import seed
from tabnanny import verbose

from traitlets import Any

import torch
from stable_baselines3 import PPO
from environment.py import CustomEnv


def create_ppo_model(env: CustomEnv, device: str = "auto") -> PPO:
    return PPO(
        policy=...,
        env=...,
        learning_rate=...,
        n_steps=...,
        batch_size=...,
        n_epochs=...,
        gamma=...,
        gae_lambda=...,
        clip_range=...,
        clip_range_vf=...,
        normalize_advantage=...,
        ent_coef=...,
        vf_coef=...,
        max_grad_norm=...,
        use_sde=...,
        sde_sample_freq=...,
        rollout_buffer_class=...,
        rollout_buffer_kwargs=...,
        target_kl=...,
        stats_window_size=...,
        tensorboard_log=...,
        policy_kwargs=...,
        verbose=...,
        seed=...,
        device=...,
        _init_setup_model=...
    )
