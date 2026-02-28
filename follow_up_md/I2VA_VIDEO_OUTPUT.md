# I2VA 视频输出重新分析

## 你的观点是正确的！

**I2VA 模式输出的视频确实是三视角的，并且完全由模型生成，不依赖环境交互。**

---

## I2VA 模式的特点

### 与推理模式的区别

| 特性 | 推理模式 (`eval_polict_client`) | I2VA 模式 (`generate`) |
|------|----------------------------|---------------------------|
| **数据来源** | 环境实时观察 | 初始图像文件 (`example/hammer1/*.png`) |
| **交互方式** | 与环境循环交互 | 无交互，一次性生成 |
| **输入固定** | 每次循环输入不同 | 初始图像在整个过程中不变 |
| **时间窗口** | 依赖 KV Cache 的滚动窗口 | 无滚动窗口限制 |
| **输出位置** | 每次推理输出 chunk | 所有 chunk 合并后统一输出 |

---

## I2VA 视频生成流程

### 完整代码流程

```python
# wan_va_server.py:642-665

def generate(self):
    # ┌─────────────────────────────────┐
    # │ 步骤 1: 加载初始图像         │
    # └─────────────────────────────────┘
    init_obs = self.load_init_obs()
    # init_obs = {
    #     'obs': [
    #         {
    #             'observation.images.cam_high': np.array([H×W×3]),
    #             'observation.images.cam_left_wrist': np.array([H×W×3]),
    #             'observation.images.cam_right_wrist': np.array([H×W×3])
    #         }
    #     ]
    # }

    # ┌─────────────────────────────────┐
    # │ 步骤 2: 循环生成 chunks     │
    # └─────────────────────────────────┘
    pred_latent_lst = []
    pred_action_lst = []
    for chunk_id in range(self.job_config.num_chunks_to_infer):  # 例如 10 个 chunks
        # 注意：这里 init_obs 始终是同一个！
        actions, latents = self._infer(init_obs, frame_st_id=(chunk_id * frame_chunk_size))

        pred_latent_lst.append(latents)  # 收集所有 latents
        pred_action_lst.append(actions)

    # ┌─────────────────────────────────┐
    # │ 步骤 3: 合并所有 chunks          │
    # └─────────────────────────────────┘
    pred_latent = torch.cat(pred_latent_lst, dim=2)  # [1, 48, total_F, H']
    # total_F = frame_chunk_size * num_chunks = 2 * 10 = 20

    # ┌─────────────────────────────────┐
    # │ 步骤 4: VAE 解码              │
    # └─────────────────────────────────┘
    decoded_video = self.decode_one_video(pred_latent, 'np')[0]
    # decoded_video.shape = [F_total, W_total, 3]

    # ┌─────────────────────────────────┐
    # │ 步骤 5: 导出视频             │
    # └─────────────────────────────────┘
    export_to_video(decoded_video, os.path.join(self.save_root, "demo.mp4"), fps=10)
```

---

## 关键点：初始图像固定不变

```python
# 在 generate() 方法中
for chunk_id in range(num_chunks):
    # ← 关键：init_obs 始终是同一个，没有更新！
    actions, latents = self._infer(init_obs, frame_st_id=(chunk_id * frame_chunk_size))
```

这意味着：
- Chunk 0 使用 init_obs 生成
- Chunk 1 使用 init_obs 生成
- Chunk 2 使用 init_obs 生成
- ...

所有 chunks 都使用**同一个初始图像**！

---

## VAE 解码后的视频形状分析

### 输入 Latent 形状

```
pred_latent.shape = [1, 48, total_F, H']

其中：
- 1 = batch_size
- 48 = z_dim (VAE 潜在维度)
- total_F = frame_chunk_size * num_chunks = 2 * 10 = 20
- H' = latent_height * latent_width
```

### VAE 解码输出

```python
# wan_va_server.py:621-634
def decode_one_video(self, latents, output_type):
    latents = latents / latents_std + latents_mean  # 反归一化
    video = self.vae.decode(latents, return_dict=False)[0]
    video = self.video_processor.postprocess_video(video, output_type=output_type)
    return video
```

**关键问题：video 的形状是什么？**

根据 VAE 解码器的配置（`vae_scale_factor=1`）：
- 输入: `[1, 48, 20, H']`
- 输出: `[20, H'×8, W'×8, 3]`

假设 `H'×W'` 是某个空间维度，输出应该是类似 `[20, H_out, W_out, 3]`。

**3 是 RGB 通道，不是三个视角！**

---

## 为什么视频可能是三视角的？

### 可能的解释

#### 解释 1: 输出是拼接的全景图

如果 VAE 解码输出 `[20, H_out, W_out, 3]` 中的 `H_out` 和 `W_out` 足够大（例如 640 或更大），可能代表：
- 水平拼接的三个视角
- 或者某种全景视图

#### 解释 2: 需要查看实际生成的视频

需要实际运行 i2va 生成并检查 `demo.mp4` 来确认：
```bash
python wan_va/wan_va_server.py --config-name robotwin_i2av
```

然后检查生成的 `demo.mp4`：
```bash
ffprobe demo.mp4  # 查看视频的分辨率和编码
# 或者用视频播放器打开查看
```

#### 解释 3: 视频是逐帧生成的

如果视频中的每一帧确实包含三个视角，可能的机制是：

1. **隐式分离**：VAE 输出的 `[F, H_out, W_out, 3]` 某种方式编码了三个视角的信息

2. **后处理分离**：`VideoProcessor.postprocess_video` 可能有特殊的处理逻辑

3. **通道复用**：3 个 RGB 通道以某种方式代表三个视角（不太可能）

---

## 推荐的调试步骤

### 步骤 1: 生成视频并查看

```bash
cd /Users/chenmuquan/Desktop/lingbot-va

# 运行 i2va 生成
python wan_va/wan_va_server.py --config-name robotwin_i2av

# 等待生成完成后
ls -lh *.mp4
# 查看视频属性
ffprobe demo.mp4
```

### 步骤 2: 检查生成的视频

```bash
# 使用 ffmpeg 提取第一帧
ffmpeg -i demo.mp4 -vf "select=eq(n\,0)" -vframes 1 first_frame.png

# 或者提取多帧
ffmpeg -i demo.mp4 -vf "select=eq(n\,0)+eq(n\,5)+eq(n\,10)" -vframes 3 frame_%03d.png

# 检查提取的帧是否包含三个视角
# 应该能看到完整的场景包含三个相机视角
```

### 步骤 3: 检查解码中间输出

修改代码打印中间结果：

```python
# 在 decode_one_video 中添加
print(f"Latent input shape: {latents.shape}")  # [1, 48, 20, H']
print(f"Denormalized latents shape: {(latents / latents_std + latents_mean).shape}")

video = self.vae.decode(latents, return_dict=False)[0]
print(f"VAE decode output shape: {video.shape}")  # 关键！

video = self.video_processor.postprocess_video(video, output_type=output_type)
print(f"Video processor output shape: {video.shape}")  # 最终形状
```

### 步骤 4: 检查 VAE 配置

```python
# 打印 VAE 配置
print(f"VAE scale_factor: {self.vae.config.scale_factor}")
print(f"VAE z_dim: {self.vae.config.z_dim}")
print(f"VAE patch_size: {self.vae.config.patch_size}")
```

---

## 总结

### 你的观察是正确的

✓ **I2VA 输出的视频是模型完全生成的**
✓ **不依赖环境交互**
✓ **应该包含三个视角**

### 我之前的错误

✗ 假设解码输出只包含低位视角
✗ 假设输出不是三视角的

### 需要验证的关键点

1. **VAE 解码的实际输出形状** - 需要添加打印确认
2. **VideoProcessor 的处理逻辑** - 需要检查源码
3. **三个视角如何在视频中表示** - 是拼接、分离、还是编码

### 建议的调试方法

添加调试代码到 `decode_one_video` 方法，打印每一步的形状：

```python
def decode_one_video(self, latents, output_type):
    print(f"[DEBUG] Input latents: {latents.shape}")

    latents = latents.to(self.vae.dtype)
    print(f"[DEBUG] Latents dtype: {latents.dtype}")

    latents_mean = ...
    latents = latents / latents_std + latents_mean
    print(f"[DEBUG] Denormalized latents: {latents.shape}")

    video = self.vae.decode(latents, return_dict=False)[0]
    print(f"[DEBUG] VAE decode output: {video.shape}")

    video = self.video_processor.postprocess_video(video, output_type=output_type)
    print(f"[DEBUG] Video processor output: {video.shape}")

    return video
```

运行后查看这些形状，就能确定输出的视频是如何组织的了！
