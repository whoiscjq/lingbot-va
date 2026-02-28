# 模型如何预测三个视角的视频

## 核心答案

**Wan2.2 模型将三个视角的视频处理为一个"多通道视频"，在 latent 空间进行统一预测和去噪。**

它不是"预测三个独立视频"，而是"预测一个包含三个视角的统一视频表示"。

---

## 完整处理流程

### 1. 输入：三个相机图像

```
原始观察 (obs['obs'])
┌─────────────────────────────────────────┐
│  [B, H, W, 3] - 每个相机一个列表 │
│                                          │
│  观察列表：[obs1, obs2, ..., obsN]    │
│                                          │
│  每个观察包含：                           │
│  {                                      │
│    'observation.images.cam_high':      [H, W, 3] │
│    'observation.images.cam_left_wrist': [H, W, 3] │
│    'observation.images.cam_right_wrist': [H, W, 3] │
│  }                                      │
└─────────────────────────────────────────────────┘
```

### 2. 视频编码：三个视角 → 统一 Latent

**关键代码** (`wan_va_server.py:322-370`)：

```python
def _encode_obs(self, obs):
    images = obs['obs']  # 历史观察列表

    videos = []
    for k_i, k in enumerate(self.job_config.obs_cam_keys):
        # 遍历三个相机
        history_video_k = torch.stack([each[k] for each in images])

        # 统一插值到相同大小
        if self.env_type == 'robotwin_tshape':
            if k_i == 0:  # camera high
                height_i, width_i = self.height, self.width  # 256×320
            else:
                height_i, width_i = self.height // 2, self.width // 2  # 128×160

        history_video_k = F.interpolate(history_video_k,
                                            size=(height_i, width_i), ...)

        videos.append(history_video_k)

    # 关键：将三个视角拼接为一个"多通道视频"
    if self.env_type == 'robotwin_tshape':
        videos_high = videos[0] / 255.0 * 2.0 - 1.0
        videos_left_and_right = torch.cat(videos[1:], dim=0) / 255.0 * 2.0 - 1.0

        enc_out_high = self.streaming_vae.encode_chunk(videos_high)
        enc_out_left_and_right = self.streaming_vae_half.encode_chunk(videos_left_and_right)

        # ← 拼接编码输出！
        enc_out = torch.cat([
            torch.cat(enc_out_left_and_right.split(1, dim=0), dim=-1),  # [B, 48, F, H']
            enc_out_high,                                            # [B, 48, F, H']
        ], dim=-2)
        #                                       [B, 48, F, H'+H', 48×2=96]
```

### 3. Latent 空间：多通道表示

```
三个视角的编码结果
┌─────────────────────────────────────────┐
│  高位相机 [B, 48, F, H']        │  48维 latent
│    ↓ 归一化                      │
│  [B, 48, F, H']                │
├─────────────────────────────────────────┤
│  左右相机 [B, 48, F, H']        │  拼接为 48×2=96 维
│    ↓ 归一化                      │
│  [B, 96, F, H']                │
└─────────────────────────────────────────┘
              ↓ 沿通道维度拼接
        ┌─────────────────────────┐
        │ [B, 144, F, H'+H'] │  ← 统一的多通道视频 latent
        │                        │
        │ 144 = 48 + 96          │
        │                        │
        └─────────────────────────┘
```

### 4. Transformer 处理：统一去噪

```python
# 在 forward 或 forward_train 中
hidden_states = [latent_hidden_states,           # 噪声视频 (3视角拼接)
              condition_latent_hidden_states,   # 干净视频 (3视角拼接)
              action_hidden_states,           # 噪声动作
              condition_action_hidden_states,  # 干净动作
              ]  # 沿序列维度拼接

# Transformer 看到的是统一的多通道序列
for block in self.blocks:
    hidden_states = block(hidden_states, text_hidden_states, ...)

# 输出也统一的多通道表示
latent_hidden_states, _, action_hidden_states, _ = torch.split(
    hidden_states, split_list, dim=1
)
```

### 5. 视频解码：Latent → 分离视角

**关键代码** (`wan_va_server.py:621-634`)：

```python
def decode_one_video(self, latents, output_type):
    # 形状: [B, 144, F, H'+H']
    # 144 = 48 (高位) + 96 (左右拼接)

    latents = latents / latents_std + latents_mean  # 反归一化

    # VAE 解码
    video = self.vae.decode(latents, return_dict=False)[0]
    # 输出形状: [F, H'+H', W', 3]

    # 后处理
    video = self.video_processor.postprocess_video(video, output_type=output_type)
```

**解码后的视频形状**：

```
Latent: [1, 144, F, H'+H']
         ↓ VAE 解码
Video:  [F, H_total, W_total, 3]
       = [F, (H'+H')×8, (W'+W')×8, 3]  # 还原到像素空间
       = [F, (32+48)×8, (40+80)×8, 3]
       = [F, 640, 640, 3]
```

但这里有个问题！解码后的视频需要**再分离回三个视角**。

让我检查解码后的处理：
```
[640, 640, 3] 中的 3 表示 RGB 通道，
不是三个视角的分离！
```

---

## 关键代码分析

### 编码时的拼接逻辑

```python
# wan_va_server.py:348-359

# 高位相机 (256×320)
videos_high = videos[0] / 255.0 * 2.0 - 1.0

# 左右相机拼接 (每个 128×320)
videos_left_and_right = torch.cat(videos[1:], dim=0)
videos_left_and_right = videos_left_and_right / 255.0 * 2.0 - 1.0

# 分别编码
enc_out_high = vae.encode_chunk(videos_high)           # [B, 48, F, H_high']
enc_out_lr = vae_half.encode_chunk(videos_left_and_right)  # [B, 48, F, H_lr']

# 拼接编码输出 (注意是 dim=-2，即通道维度)
enc_out = torch.cat([
    torch.cat(enc_out_lr.split(1, dim=0), dim=-1),  # [B, 96, F, H_lr']
    enc_out_high,                                         # [B, 48, F, H_high']
], dim=-2)
```

**结果形状**: `[B, 48+96, F, H'+H']` = `[B, 144, F, H_total]`

### 实际的多视角表示方式

**Wan2.2 采用的是"通道拼接"方式**：

```
视角布局：
┌─────────────────────────────────┐
│  高位相机 (256×320)          │ → 编码为 48 通道
├─────────────────────────────────┤
│  左相机 (128×160)           │ │
│  右相机 (128×160)           │ │ → 编码后拼接为 96 通道
└─────────────────────────────────┘
              ↓
    [B, 144, F, H_total]  ← 统一的 latent
    48 (高位) + 96 (左右×2)
```

---

## 解码后的视频问题

### 问题：如何分离三个视角？

解码后的 `[F, 640, 640, 3]` 中：
- **3 表示 RGB 通道**（红绿蓝）
- **不是三个视角的分离**

这意味着**解码器没有区分视角的显式结构**！

### 可能的解释

1. **模型训练时没有学习分离**：解码器学习的是从 144 通道 latent 到 640×640 RGB 图像的映射
2. **视角信息混合在 latent 中**：三个视角的信息融合在一起
3. **输出是"全景式"视频**：可能不是分离的三个视角，而是某种全景或拼接

---

## 总结

### Wan2.2 如何处理三个视角

| 阶段 | 处理方式 | 形状变化 |
|------|---------|-----------|
| **输入** | 3 个独立的相机图像 | `[3个相机]` 每个 `[H, W, 3]` |
| **编码** | 分别编码，拼接 latent | 高位: `[B, 48, F, H']` <br> 左右: `[B, 96, F, H']` |
| **合并** | 沿通道维度拼接 | `[B, 144, F, H_total]` |
| **去噪** | Transformer 统一处理 | `[B, 144, F, H_total]` |
| **解码** | VAE 统一解码 | `[F, 640, 640, 3]` |

### 关键要点

1. **不是预测三个独立视频**，而是预测一个**包含多视角信息的统一表示**

2. **视角间通过注意力交互**：在 latent 空间，三个视角的信息可以互相影响

3. **解码输出可能不是分离的**：输出是 RGB 图像，不是三个独立的视角

4. **144 维度 = 48 + 96**：高位相机的 48 维 + 左右相机的 48×2 维

### 为什么这样设计？

这种设计的优势：
- **统一处理**：一个 Transformer 可以同时处理所有视角
- **视角间交互**：允许模型学习视角间的对应关系
- **高效**：避免分别处理三个视角的开销

代价：
- **视角分离模糊**：解码后难以明确分离三个视角
- **计算复杂度高**：144 维度比单一视角的 48 维度高
