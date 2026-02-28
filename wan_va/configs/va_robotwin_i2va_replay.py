# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_robotwin_cfg import va_robotwin_cfg

va_robotwin_i2va_replay_cfg = EasyDict(__name__='Config: VA robotwin i2va')
va_robotwin_i2va_replay_cfg.update(va_robotwin_cfg)

va_robotwin_i2va_replay_cfg.input_img_path = 'example/robotwin'
va_robotwin_i2va_replay_cfg.num_chunks_to_infer = 10
va_robotwin_i2va_replay_cfg.prompt = 'pick up the black water cup, and put it beside the blue block'
va_robotwin_i2va_replay_cfg.infer_mode = 'replay'
va_robotwin_i2va_replay_cfg.latent_folder = '/mnt/shared-storage-user/chenjunqi/lingbot-va/train_out/real/Grab the medium-sized white mug, rotate it, and put it upside down on the table._20260223_080328' 