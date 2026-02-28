# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_robotwin_cfg import va_robotwin_cfg

va_robotwin_i2va_cfg = EasyDict(__name__='Config: VA robotwin i2va')
va_robotwin_i2va_cfg.update(va_robotwin_cfg)

va_robotwin_i2va_cfg.input_img_path = 'example/hammer1'
va_robotwin_i2va_cfg.num_chunks_to_infer = 10
va_robotwin_i2va_cfg.prompt = 'pick up the black water cup, and put it beside the blue block'
va_robotwin_i2va_cfg.infer_mode = 'i2va'