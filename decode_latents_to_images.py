#!/usr/bin/env python3
"""
从 latents.pt 文件解码并保存为图片

使用方法:
    python decode_latents_to_images.py --latents_path real/exp_name/latents_0.pt --output_dir decoded_images
"""

import torch
import argparse
import os
from PIL import Image
import numpy as np
from hytra import load_config
from wan_va.wan_va_server import WanVADiffusionServer


def parse_args():
    parser = argparse.ArgumentParser(description='Decode latents to images')
    parser.add_argument('--config', type=str, default='va_robotwin_i2va',
                       help='Config name (default: va_robotwin_i2va)')
    parser.add_argument('--latents_path', type=str, required=True,
                       help='Path to latents.pt file or directory containing multiple latents')
    parser.add_argument('--output_dir', type=str, default='decoded_images',
                       help='Directory to save decoded images')
    parser.add_argument('--image_format', type=str, default='png',
                       choices=['png', 'jpg'], help='Output image format')
    parser.add_argument('--save_video', action='store_true',
                       help='Also save as video file')
    parser.add_argument('--video_fps', type=int, default=10,
                       help='Video frame rate (default: 10)')
    return parser.parse_args()


def load_server(config_name='va_robotwin_i2va'):
    """加载 WanVA server"""
    print(f"[INFO] Loading config: {config_name}")
    config = load_config(config_name)
    server = WanVADiffusionServer(
        config,
        'robotwin_tshape',
        'wan_va',
        None
    )
    return server


def decode_single_latent(server, latents_path, output_dir, image_format='png'):
    """解码单个 latents 文件"""
    print(f"\n{'='*60}")
    print(f"[INFO] Loading latents from: {latents_path}")

    # 加载 latents
    latents = torch.load(latents_path, map_location='cpu')
    print(f"[INFO] Latents shape: {latents.shape}")
    print(f"[INFO] Latents dtype: {latents.dtype}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 解码为视频帧
    print(f"[INFO] Decoding latents to video frames...")
    video_frames = server.decode_one_video(latents, 'np')
    print(f"[INFO] Decoded video shape: {video_frames.shape}")
    print(f"[INFO] Video dtype: {video_frames.dtype}")
    print(f"[INFO] Value range: [{video_frames.min():.2f}, {video_frames.max():.2f}]")

    # 保存每一帧
    base_name = os.path.splitext(os.path.basename(latents_path))[0]
    num_frames = video_frames.shape[0]

    print(f"\n[INFO] Saving {num_frames} frame(s) to {output_dir}/...")

    for i in range(num_frames):
        frame = video_frames[i]

        # 转换为 uint8 (0-255)
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)

        # 保存图片
        output_path = os.path.join(output_dir, f"{base_name}_frame_{i:03d}.{image_format}")
        img = Image.fromarray(frame)
        img.save(output_path)
        print(f"  [Saved] {output_path}")

    return video_frames


def decode_multiple_latents(server, latents_dir, output_dir, image_format='png', save_video=False, video_fps=10):
    """解码多个 latents 文件并合并"""
    print(f"\n{'='*60}")
    print(f"[INFO] Processing directory: {latents_dir}")

    # 查找所有 latents_*.pt 文件
    latents_files = sorted([
        f for f in os.listdir(latents_dir)
        if f.startswith('latents_') and f.endswith('.pt')
    ])

    if not latents_files:
        print(f"[ERROR] No latents_*.pt files found in {latents_dir}")
        return None

    print(f"[INFO] Found {len(latents_files)} latents files: {latents_files}")

    # 加载并拼接所有 latents
    all_latents = []
    for latents_file in latents_files:
        latents_path = os.path.join(latents_dir, latents_file)
        print(f"\n[INFO] Loading {latents_file}...")
        latents = torch.load(latents_path, map_location='cpu')
        print(f"  Shape: {latents.shape}")
        all_latents.append(latents)

    # 沿帧维度拼接
    print(f"\n[INFO] Concatenating all latents along frame dimension...")
    concatenated_latents = torch.cat(all_latents, dim=2)
    print(f"[INFO] Concatenated shape: {concatenated_latents.shape}")

    # 解码为视频
    print(f"\n[INFO] Decoding to video frames...")
    video_frames = server.decode_one_video(concatenated_latents, 'np')
    print(f"[INFO] Decoded video shape: {video_frames.shape}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存每一帧
    num_frames = video_frames.shape[0]
    print(f"\n[INFO] Saving {num_frames} frame(s) to {output_dir}/...")

    for i in range(num_frames):
        frame = video_frames[i]

        # 转换为 uint8 (0-255)
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)

        # 保存图片
        output_path = os.path.join(output_dir, f"frame_{i:03d}.{image_format}")
        img = Image.fromarray(frame)
        img.save(output_path)

        if (i + 1) % 10 == 0:
            print(f"  [Progress] Saved {i+1}/{num_frames} frames")

    # 保存为视频
    if save_video:
        video_path = os.path.join(output_dir, "decoded_video.mp4")
        print(f"\n[INFO] Saving video to {video_path}...")

        try:
            import imageio
            imageio.mimwrite(video_path, video_frames, fps=video_fps)
            print(f"  [Saved] {video_path}")
        except ImportError:
            print(f"  [Warning] imageio not installed, skipping video save")
            print(f"  [Hint] Install with: pip install imageio")

    return video_frames


def main():
    args = parse_args()

    print("="*60)
    print("Latents to Images Decoder")
    print("="*60)

    # 加载 server
    server = load_server(args.config)

    # 判断是文件还是目录
    latents_path = args.latents_path
    if os.path.isfile(latents_path):
        # 单个文件
        decode_single_latent(server, latents_path, args.output_dir, args.image_format)
    elif os.path.isdir(latents_path):
        # 目录（多个文件）
        decode_multiple_latents(
            server, latents_path, args.output_dir,
            args.image_format, args.save_video, args.video_fps
        )
    else:
        print(f"[ERROR] Path not found: {latents_path}")

    print(f"\n{'='*60}")
    print(f"[DONE] Images saved to: {args.output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
