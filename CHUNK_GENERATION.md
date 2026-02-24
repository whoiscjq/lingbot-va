# 为什么每个 chunk 都不一样

## 核心答案

**你说得对！每个 chunk 确实应该不一样！**

我之前的理解有误。让我重新分析代码逻辑。

---

## 关键代码分析

### 去噪循环中的条件处理

**位置**: `wan_va_server.py:483-517` 和 `_prepare_latent_input:262-293`

```python
# 在 _infer 中，视频去噪循环
for i, t in enumerate(timesteps):
    last_step = i == len(timesteps) - 1

    # 关键：latent_cond 的使用
    if frame_st_id == 0:  # ← 只有第一个 chunk！
        latent_cond = init_latent[:, :, 0:1].to(self.dtype)
    else:
        latent_cond = None  # ← 其他 chunks 都是 None！

    # 准备输入
    input_dict = self._prepare_latent_input(
        latents,              # 随机噪声（每次循环都在变）
        None,
        t,
        t,
        latent_cond,         # ← 第一个 chunk 用条件，其他不用
        None,
        frame_st_id=frame_st_id
    )
```

### `_prepare_latent_input` 中的条件注入

**位置**: `wan_va_server.py:289-292`

```python
def _prepare_latent_input(..., latent_cond=None, ...):
    input_dict = dict()

    if latent_model_input is not None:
        input_dict['latent_res_lst'] = {
            'noisy_latents': latent_model_input,
            'timesteps': ...,
            'grid_id': ...,
            'text_emb': ...,
        }

    # 关键：如果有条件，用条件替换第一帧
    if latent_cond is not None:
        input_dict['latent_res_lst']['noisy_latents'][:, :, 0:1] = latent_cond[:, :, 0:1]
        input_dict['latent_res_lst']['timesteps'][0:1] *= 0  # ← 时间步设为 0！
```

---

## 每个 Chunk 的处理差异

### Chunk 0 (frame_st_id = 0)

```
┌─────────────────────────────────┐
│ 初始化: init_latent = _encode_obs(init_obs)        │
│                                        │
│ 去噪循环:                                   │
│  for t in timesteps:                         │
│    ├─ latent_cond = init_latent[:, :, 0:1]     │ ← 使用初始 latent 作为条件
│    ├─ noisy_latents[:, :, 0:1] = init_latent    │ ← 第 0 帧用干净 latent
│    └─ timesteps[0] *= 0                       │ ← 第 0 帧 t=0
│                                        │
│  输出: 基于初始图像，逐步生成后续帧     │
└─────────────────────────────────┘
```

**结果**：生成一个从初始图像逐步发展的视频序列。

### Chunk 1, 2, 3, ... (frame_st_id > 0)

```
┌─────────────────────────────────┐
│ 初始化: init_latent = 初始值           │
│                                        │
│ 去噪循环:                                   │
│  for t in timesteps:                         │
│    ├─ latent_cond = None                    │ ← 不使用条件
│    └─ noisy_latents = 纯随机噪声         │ ← 每帧都是噪声
│                                        │
│  输出: 从纯噪声生成视频                  │
└─────────────────────────────────┘
```

**结果**：每个 chunk 独立地从噪声生成视频，相互之间没有延续性！

---

## 完整流程图

```
初始图像 (example/hammer1/*.png)
    ↓
VAE 编码 → init_latent [1, 48, 1, H']
    ↓
    ┌─────────────────────┬─────────────────┐
    ↓                    │                    │
Chunk 0:           Chunk 1:    Chunk 2:    ...
    ↓                    ↓                    ↓
有条件的去噪       纯随机噪声      纯随机噪声  纯随机噪声
    ↓                    ↓                    ↓
基于初始图像的逐步   独立生成      独立生成    ...
    ↓                    ↓                    ↓
    │                    │                    │
    └─────────────────────┴─────────────────────┘
    ↓
    ↓                    ↓                    ↓
拼接所有 chunks → [1, 48, total_F, H']
    ↓
VAE 解码 → 视频输出 [total_F, H, W, 3]
```

---

## 为什么这样设计？

### Chunk 0：延续性生成

- **目的**：从初始图像生成连贯的视频序列
- **机制**：使用 `init_latent` 作为条件
- **时间步**：第一帧 t=0（干净），后续帧逐步增加噪声
- **类比**：类似扩散模型的标准采样过程

### Chunk 1, 2, ...：独立生成

- **目的**：每个 chunk 独立生成一段视频
- **机制**：不使用任何条件，纯从噪声生成
- **时间步**：所有帧都使用相同的时间步范围
- **类比**：类似从纯噪声开始的采样

### 为什么要这样？

**可能的原因**：

1. **生成多样性**：不同的 chunk 生成不同的变体
2. **容错性**：即使某些 chunk 失败，其他仍然有效
3. **并行友好**：理论上每个 chunk 可以独立处理

**但是**：
- 这会导致视频不连贯
- Chunk 1, 2, ... 之间没有逻辑关联

---

## 预期的行为

### Chunk 0 的输出

应该是一个连贯的视频序列：
- Frame 0: 接近初始图像
- Frame 1: 在 Frame 0 基础上发展
- Frame 2: 在 Frame 1 基础上发展
- ...

类似于从初始图像开始的标准视频扩散生成。

### Chunk 1, 2, ... 的输出

每个 chunk 都是独立的，从纯噪声生成：
- 没有明确的起点
- 相互之间不连贯
- 可能看起来完全不同

---

## 问题：这样的设计合理吗？

### 可能是 Bug？

如果设计意图是生成一个连贯的完整视频，那么：

1. **Chunk 1, 2, ... 应该也使用条件** - 继承之前 chunk 的输出
2. **或者所有 chunks 应该是纯随机** - 没有 Chunk 0 的特殊处理

### 可能是预期行为？

如果这是有意设计，那么：

1. **Chunk 0** 生成"基础"或"参考"版本
2. **其他 chunks** 生成"变体"或"多样化"版本
3. 最终可能选择最好的版本进行后处理

---

## 验证方法

### 添加调试代码

修改 `generate()` 方法，打印每个 chunk 的信息：

```python
def generate(self):
    pred_latent_lst = []
    for chunk_id in range(self.job_config.num_chunks_to_infer):
        print(f"\n=== Chunk {chunk_id} ===")
        print(f"frame_st_id = {chunk_id * self.job_config.frame_chunk_size}")

        if chunk_id == 0:
            print(f"Using latent_cond (conditioned generation)")
            print(f"init_latent shape: {self.init_latent.shape}")
        else:
            print(f"No latent_cond (unconditioned generation)")

        actions, latents = self._infer(init_obs, frame_st_id=(...))
        print(f"Output latents shape: {latents.shape}")

        pred_latent_lst.append(latents)
```

### 检查输出视频

```bash
# 生成视频
python wan_va/wan_va_server.py --config-name robotwin_i2av

# 提取帧并对比不同 chunk
ffmpeg -i demo.mp4 -vf "select=eq(n\,0)" -vframes 1 chunk0_frame.png
ffmpeg -i demo.mp4 -vf "select=eq(n\,5)" -vframes 1 chunk1_frame.png
ffmpeg -i demo.mp4 -vf "select=eq(n\,10)" -vframes 1 chunk2_frame.png
```

如果 Chunk 1, 2, ... 的帧看起来完全不同（都是噪声），那么确认了问题。

---

## 总结

你的观察是正确的：**使用同一个初始图像，但每个 chunk 的处理方式不同**。

- **Chunk 0**：条件生成（有初始 latent 作为条件）
- **Chunk 1, 2, ...**：无条件生成（纯从噪声）

这会导致 Chunk 0 生成连贯视频，而其他 chunks 生成不连贯的视频片段。如果这是一个 bug，需要修复；如果是有意设计，那么这是一个奇怪的设计决策。
