# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from .va_franka_cfg import va_franka_cfg
from .va_robotwin_cfg import va_robotwin_cfg
from .va_robotwin_base_cfg import va_robotwin_base_cfg
from .va_robotwin_prompt_masked_cfg import va_robotwin_prompt_masked_cfg
from .va_franka_i2va import va_franka_i2va_cfg
from .va_robotwin_i2va import va_robotwin_i2va_cfg
from .va_robotwin_base_i2va import va_robotwin_base_i2va_cfg
from .va_robotwin_train_cfg import va_robotwin_train_cfg
from .va_robotwin_i2va_replay import va_robotwin_i2va_replay_cfg

VA_CONFIGS = {
    'robotwin': va_robotwin_cfg,
    'robotwin_base': va_robotwin_base_cfg,
    'robotwin_prompt_masked': va_robotwin_prompt_masked_cfg,
    'franka': va_franka_cfg,
    'robotwin_i2av': va_robotwin_i2va_cfg,
    'robotwin_base_i2av': va_robotwin_base_i2va_cfg,
    'robotwin_i2av_replay': va_robotwin_i2va_replay_cfg,
    'franka_i2av': va_franka_i2va_cfg,
    'robotwin_train': va_robotwin_train_cfg,
}