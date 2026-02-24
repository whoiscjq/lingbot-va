# 模型推理过程 Roadmap

## 概览

本文档提供理解和阅读 `lingbot-va` 模型推理过程的路线图，从代码入口到最终输出。

---

## 第一阶段：环境准备

### 文件位置
```
evaluation/robotwin/eval_polict_client_openpi.py (主评估脚本)
  ↓ 调用
evaluation/robotwin/websocket_client_policy.py (WebSocket 客户端)
  ↓ 连接
wan_va/wan_va_server.py (推理服务器)
  ↓ 使用
wan_va/modules/model.py (Transformer 模型)
wan_va/modules/utils.py (工具函数)
wan_va/configs/va_robotwin_cfg.py (配置文件)
```

### 关键配置文件

**`wan_va/configs/va_robotwin_cfg.py`** - 核心配置
```python
# 相机配置
obs_cam_keys = [
    'observation.images.cam_high',      # 高位相机: 256×320
    'observation.images.cam_left_wrist',  # 左手腕相机: 128×160
    'observation.images.cam_right_wrist'  # 右手腕相机: 128×160
]

# 动作配置
action_dim = 30               # 动作维度
action_per_frame = 16          # 每帧动作数

# 推理配置
num_inference_steps = 25        # 视频去噪步数
action_num_inference_steps = 50  # 动作去噪步数
frame_chunk_size = 2            # 每次推理的帧数

# 环境类型
env_type = 'robotwin_tshape'     # 特殊模式：高低分辨率相机混合
```

---

## 第二阶段：推理循环主流程

### 入口点：`eval_polict_client_openpi.py:554-608`

```python
while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
    # ┌─────────────────────────────────────────┐
    # │ 步骤 1: 推理生成动作          │
    # └─────────────────────────────────────────┘
    ret = model.infer(dict(obs=first_obs, prompt=prompt, ...))
    action = ret['action']

    # ┌─────────────────────────────────────────┐
    # │ 步骤 2: 执行动作序列          │
    # └─────────────────────────────────────────┘
    for i in range(action.shape[1]):
        for j in range(action.shape[2]):
            TASK_ENV.take_action(ee_action, action_type='ee')

    # ┌─────────────────────────────────────────┐
    # │ 步骤 3: 更新 KV Cache             │
    # └─────────────────────────────────────────┘
    model.infer(dict(obs=key_frame_list, compute_kv_cache=True, ...))

    # 重复，直到任务完成或达到步数限制
```

**关键循环变量**：
- `frame_st_id` - 当前推理帧索引
- `first_obs` - 当前帧的观察
- `key_frame_list` - 关键帧列表（用于更新 cache）

---

## 第三阶段：服务器端推理详解

### 3.1 重置阶段 - `_reset()`

**位置**: `wan_va_server.py:372-436`

```python
def _reset(self, prompt=None):
    # ┌─────────────────────────────────────────┐
    # │ 清空 KV Cache                  │
    # └─────────────────────────────────────────┘
    self.transformer.clear_cache(self.cache_name)
    self.streaming_vae.clear_cache()

    # ┌─────────────────────────────────────────┐
    # │ 初始化缓存参数                 │
    # └─────────────────────────────────────────┘
    self.frame_st_id = 0
    self.action_per_frame = 16

    # ┌─────────────────────────────────────────┐
    # │ 编码初始观察                   │
    # └─────────────────────────────────────────┘
    if frame_st_id == 0:
        init_latent = self._encode_obs(obs)
        self.init_latent = init_latent

    # ┌─────────────────────────────────────────┐
    # │ 编码文本提示词                 │
    # └─────────────────────────────────────────┘
    self.prompt_embeds, self.negative_prompt_embeds = self.encode_prompt(prompt, ...)

    # ┌─────────────────────────────────────────┐
    # │ 创建保存目录                    │
    # └─────────────────────────────────────────┘
    self.exp_save_root = ...
```

---

### 3.2 观察编码 - `_encode_obs()`

**位置**: `wan_va_server.py:322-370`

```python
def _encode_obs(self, obs):
    images = obs['obs']  # 历史观察列表: [obs1, obs2, ..., obsN]

    # ┌─────────────────────────────────────────┐
    # │ 遍历三个相机                   │
    # └─────────────────────────────────────────┘
    videos = []
    for k_i, k in enumerate(self.job_config.obs_cam_keys):
        history_video_k = torch.stack([each[k] for each in images])

        # 调整到相同大小
        if k_i == 0:  # camera high (256×320)
            height_i, width_i = 256, 320
        else:  # camera left/right (128×160)
            height_i, width_i = 128, 160

        history_video_k = F.interpolate(history_video_k, size=(height_i, width_i))
        videos.append(history_video_k)

    # ┌─────────────────────────────────────────┐
    # │ 分别编码三个视角               │
    # └─────────────────────────────────────────┘
    if self.env_type == 'robotwin_tshape':
        # 高位相机
        videos_high = videos[0] / 255.0 * 2.0 - 1.0
        enc_out_high = self.streaming_vae.encode_chunk(videos_high)

        # 左右相机拼接后编码
        videos_left_and_right = torch.cat(videos[1:], dim=0)
        videos_left_and_right /= 255.0 * 2.0 - 1.0
        enc_out_lr = self.streaming_vae_half.encode_chunk(videos_left_and_right)

        # 拼接编码输出
        enc_out = torch.cat([
            torch.cat(enc_out_lr.split(1, dim=0), dim=-1),  # [B, 96, F, H']
            enc_out_high,                                            # [B, 48, F, H']
        ], dim=-2)  # [B, 144, F, H'+H']

    # ┌─────────────────────────────────────────┐
    # │ 归一化处理                       │
    # └─────────────────────────────────────────┘
    mu, logvar = torch.chunk(enc_out, 2, dim=1)
    latents_mean = ...
    latents_std = ...
    mu_norm = self.normalize_latents(mu, latents_mean, 1.0 / latents_std)
    video_latent = torch.cat(mu_norm.split(1, dim=0), dim=-1)

    return video_latent  # [B, 144, F, H']
```

**关键数据流**：
```
三个相机图像
    ↓ 插值到统一分辨率
[256×320] + [128×160] + [128×160]
    ↓ VAE 编码
[48通道] + [96通道]
    ↓ 归一化
[48通道] + [96通道] (归一化)
    ↓ 沿通道拼接
[B, 144, F, H_total] (统一 latent)
```

---

### 3.3 去噪推理 - `_infer()`

**位置**: `wan_va_server.py:438-565`

```python
def _infer(self, obs, frame_st_id=0):
    # ┌─────────────────────────────────────────┐
    # │ 初始化随机噪声                   │
    # └─────────────────────────────────────────┘
    latents = torch.randn(1, 48, frame_chunk_size, H', W')  # 视频噪声
    actions = torch.randn(1, 30, frame_chunk_size, action_per_frame, 1)  # 动作噪声

    # ┌─────────────────────────────────────────┐
    # │ 步骤 1: 视频去噪循环           │
    # └─────────────────────────────────────────┘
    for i, t in enumerate(timesteps):
        # 准备输入
        input_dict = {
            'latent_res_lst': {
                'noisy_latents': latents,
                'timesteps': t,
                'text_emb': prompt_embeds,
                ...
            }
        }

        # Transformer 去噪
        video_noise_pred = self.transformer(..., action_mode=False)
        latents = scheduler.step(video_noise_pred, t, latents)

    # ┌─────────────────────────────────────────┐
    # │ 步骤 2: 动作去噪循环           │
    # └─────────────────────────────────────────┘
    for i, t in enumerate(action_timesteps):
        # 准备输入
        input_dict = {
            'action_res_lst': {
                'noisy_latents': actions,
                'timesteps': t,
                'text_emb': prompt_embeds,
                ...
            }
        }

        # Transformer 去噪
        action_noise_pred = self.transformer(..., action_mode=True)
        actions = action_scheduler.step(action_noise_pred, t, actions)

    # 保存
    save_async(latents, f'latents_{frame_st_id}.pt')
    save_async(actions, f'actions_{frame_st_id}.pt')

    return actions, latents
```

**去噪循环详解**：
```
时间步 t: 1000 → 999 → ... → 0
    ↓
[噪声 latents] → Transformer → [噪声预测]
    ↓
Scheduler.step: latents - alpha * noise_pred
    ↓
[去噪 latents]
```

---

### 3.4 KV Cache 更新 - `_compute_kv_cache()`

**位置**: `wan_va_server.py:567-599`

```python
def _compute_kv_cache(self, obs):
    # ┌─────────────────────────────────────────┐
    # │ 编码真实的观察和动作           │
    # └─────────────────────────────────────────┘
    latent_model_input = self._encode_obs(obs)
    action_model_input = self.preprocess_action(obs['state'])

    # 准备输入
    input_dict = self._prepare_latent_input(
        latent_model_input,
        action_model_input,
        frame_st_id=self.frame_st_id
    )

    # ┌─────────────────────────────────────────┐
    # │ 存入 KV Cache (update_cache=2)  │
    # └─────────────────────────────────────────┘
    with torch.no_grad():
        self.transformer(..., update_cache=2, ...)  # 视频
        self.transformer(..., update_cache=2, ...)  # 动作

    self.frame_st_id += latent_model_input.shape[2]
```

**KV Cache 存储的是什么**：
```
真实观察 (t=0)
    ↓ 编码
[B, 144, F, H']  ← 存入 Cache 的 Key/Value

真实动作 (t=0)
    ↓ 编码
[B, L, 3072]  ← 存入 Cache 的 Key/Value
```

---

## 第四阶段：Transformer 模型内部

### 4.1 模型入口 - `forward()`

**位置**: `wan_va/modules/model.py:800-884`

```python
def forward(self, input_dict, update_cache=0, cache_name="pos",
            action_mode=False, train_mode=False):

    if train_mode:
        return self.forward_train(input_dict)

    # ┌─────────────────────────────────────────┐
    # │ 输入嵌入                       │
    # └─────────────────────────────────────────┘
    if action_mode:  # 动作分支
        latent_hidden_states = rearrange(input_dict['noisy_latents'], 'b c f h w -> b (f h w) c')
        latent_hidden_states = self.action_embedder(latent_hidden_states)
    else:  # 视频分支
        latent_hidden_states = rearrange(input_dict['noisy_latents'], ...)
        latent_hidden_states = self.patch_embedding_mlp(latent_hidden_states)

    # 文本嵌入
    text_hidden_states = self.condition_embedder.text_embedder(input_dict["text_emb"])

    # 位置编码
    latent_grid_id = input_dict['grid_id']
    rotary_emb = self.rope(latent_grid_id)[:, :, None]

    # 时间嵌入
    latent_time_steps = torch.repeat_interleave(input_dict['timesteps'], ...)
    temb, timestep_proj = self.condition_embedder(latent_time_steps, ...)

    # ┌─────────────────────────────────────────┐
    # │ Transformer Blocks 处理           │
    # └─────────────────────────────────────────┘
    for block in self.blocks:
        latent_hidden_states = block(
            latent_hidden_states,
            text_hidden_states,
            timestep_proj,
            rotary_emb,
            update_cache=update_cache,
            cache_name=cache_name
        )

    # ┌─────────────────────────────────────────┐
    # │ 输出投影                       │
    # └─────────────────────────────────────────┘
    temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
    shift, scale = rearrange(temb_scale_shift_table, 'b l n c -> b n l c').chunk(2, dim=1)

    latent_hidden_states = (self.norm_out(latent_hidden_states.float()) *
                            (1. + scale) + shift).type_as(latent_hidden_states)

    if action_mode:
        latent_hidden_states = self.action_proj_out(latent_hidden_states)
    else:
        latent_hidden_states = self.proj_out(latent_hidden_states)
        latent_hidden_states = rearrange(latent_hidden_states, 'b l (n c) -> b (l n) c', ...)

    return latent_hidden_states
```

---

### 4.2 注意力机制 - `WanAttention.forward()`

**位置**: `wan_va/modules/model.py:414-465`

```python
def forward(self, q, k, v, rotary_emb, update_cache=0, cache_name="pos"):
    # ┌─────────────────────────────────────────┐
    # │ 线性投影 Q, K, V              │
    # └─────────────────────────────────────────┘
    query, key, value = self.to_q(q), self.to_k(k), self.to_v(v)
    query = self.norm_q(query)
    key = self.norm_k(key)
    query, key, value = [unflatten(2, (self.heads, -1)), ...]

    # ┌─────────────────────────────────────────┐
    # │ 旋转位置编码                   │
    # └─────────────────────────────────────────┘
    if rotary_emb is not None:
        query = apply_rotary_emb(query, rotary_emb)
        key = apply_rotary_emb(key, rotary_emb)

    # ┌─────────────────────────────────────────┐
    # │ KV Cache 处理                   │
    # └─────────────────────────────────────────┘
    if kv_cache is not None and kv_cache['k'] is not None:
        slots = self.update_cache(cache_name, key, value, is_pred=(update_cache == 1))

        # 使用缓存的 K, V
        key_pool = self.attn_caches[cache_name]['k'][:, slots]
        value_pool = self.attn_caches[cache_name]['v'][:, slots]
        mask = self.attn_caches[cache_name]['mask']
        valid = mask.nonzero(as_tuple=False).squeeze(-1)
        key = key_pool[:, valid]
        value = value_pool[:, valid]

    # ┌─────────────────────────────────────────┐
    # │ 注意力计算                       │
    # └─────────────────────────────────────────┘
    hidden_states = self.attn_op(query, key, value)

    # ┌─────────────────────────────────────────┐
    # │ 缓存恢复（如果不是更新）          │
    # └─────────────────────────────────────────┘
    if update_cache == 0:
        if kv_cache is not None:
            self.restore_cache(cache_name, slots)

    # 输出
    hidden_states = hidden_states.flatten(2, 3)
    hidden_states = hidden_states.type_as(query)
    hidden_states = self.to_out[0](hidden_states)
    hidden_states = self.to_out[1](hidden_states)
    return hidden_states
```

---

### 4.3 注意力掩码 - `FlexAttnFunc._get_mask_mod()`

**位置**: `wan_va/modules/model.py:154-201`

```python
def _get_mask_mod(seq_ids, frame_ids, noise_ids, window_size):
    # 定义各种掩码函数

    def seq_mask(b, h, q_idx, kv_idx):
        return (seq_ids[q_idx] == seq_ids[kv_idx]) & (seq_ids[q_idx] >= 0)

    def block_causal_mask(b, h, q_idx, kv_idx):
        return (frame_ids[kv_idx] <= frame_ids[q_idx])

    def block_window_mask(b, h, q_idx, kv_idx, window_size: int):
        return ((frame_ids[q_idx] - frame_ids[kv_idx]).abs() <= window_size)

    # 组合掩码
    mask_list = [
        and_masks(clean2clean_mask, block_causal_mask),      # 干净 token 可以看未来
        and_masks(noise2clean_mask, block_causal_mask_exclude_self),  # 噪声 token 可以看干净，但不看自己
        and_masks(noise2noise_mask, block_self_mask),          # 噪声 token 只看自己
    or_masks(*mask_list)                                     # 或关系
    mask = and_masks(mask, seq_mask, block_window_mask)          # 加上序列掩码

    return mask
```

**掩码规则**：
- `clean2clean`: 干净 token 可以看未来
- `noise2clean`: 噪声 token 可以看干净（不包含自己）
- `noise2noise`: 噪声 token 只看自己和过去的噪声
- `block_window`: 限制在时间窗口内

---

## 第五阶段：输出和保存

### 5.1 动作后处理 - `postprocess_action()`

**位置**: `wan_va/wan_va_server.py:239-249`

```python
def postprocess_action(self, action):
    action = action.cpu()  # [B, C, F, H, W]

    # 取消 padding
    action = action[0, ..., 0]  # [C, F, H]

    # 反归一化
    if self.action_norm_method == 'quantiles':
        action = (action + 1) / 2 * (self.actions_q99 - self.actions_q01 + 1e-6) + self.actions_q01

    # 反转通道选择
    action = action.squeeze(0).detach().cpu().numpy()
    action = action[self.job_config.used_action_channel_ids]

    return action  # [C, F, H] 例如 [30, 2, 16]
```

---

### 5.2 视频生成 - `save_comparison_video()`

**位置**: `evaluation/robotwin/eval_polict_client_openpi.py:194-256`

```python
def save_comparison_video(real_obs_list, imagined_video, action_history, save_path, fps=15):
    # real_obs_list: 真实观察列表（包含三个相机）
    # imagined_video: 想象视频（当前为 None）
    # action_history: 动作历史（用于可视化）

    for i in range(n_frames):
        obs = real_obs_list[i]
        cam_high = obs["observation.images.cam_high"]      # 256×320×3
        cam_left = obs["observation.images.cam_left_wrist"]  # 128×160×3
        cam_right = obs["observation.images.cam_right_wrist"]  # 128×160×3

        # 调整到相同高度并水平拼接
        row_real = np.hstack([
            resize_h(cam_high, base_h),
            resize_h(cam_left, base_h),
            resize_h(cam_right, base_h)
        ])

        # 添加标题
        row_real = add_title_bar(row_real, "Real Observation (High / Left / Right)")

        # 拼接想象视频（如果存在）
        if imagined_video is not None and i < len(imagined_video):
            row_imagined = resize_h(imagined_video[i], target_width, h)
            row_imagined = add_title_bar(row_imagined, "Imagined Video Stream")
        else:
            row_imagined = np.zeros((300, target_width, 3))
            cv2.putText(row_imagined, "Coming soon", ...)

        full_frame = np.vstack([row_real, row_imagined])
        final_frames.append(full_frame)

    imageio.mimsave(save_path, final_frames, fps=fps)
```

**视频布局**：
```
┌─────────────────────────────────────────┐
│  真实观察（3个相机并排）         │
├─────────────────────────────────────────┤
│  想象视频（目前为空）             │
└─────────────────────────────────────────┘
```

---

## 第六阶段：数据流向总结

### 完整数据流图

```
┌─────────────────────────────────────────────────────────────────┐
│                       环境观察                          │
│               [obs_high, obs_left, obs_right]             │
│                    ↓                                   │
│              ┌─────────────────────────────┐               │
│              │  观察编码 _encode_obs │               │
│              │  [3相机] → [B,144,F,H']  │               │
│              └─────────────────────────────┘               │
│                    ↓                                   │
│         ┌───────────────────────────────┐                   │
│         │  视频去噪循环 _infer │                   │
│         │  [随机噪声]          │                   │
│         │  [timesteps] (N个)  │                   │
│         │  ↓                   │                   │
│         │  [去噪 latents]  ←────┐            │                   │
│         │              │ Transformer  │            │                   │
│         │              │    ↓         │            │                   │
│         │  [噪声预测]    │    scheduler │            │                   │
│         └───────────────────┘         │            │                   │
│                    ↓                                   │
│              ┌─────────────────────────────┐               │
│              │ 动作去噪循环 │               │
│         │  [随机噪声]          │                   │
│         │  [timesteps] (M个)  │                   │
│         │  ↓                   │                   │
│         │  [去噪 actions]  ←────┐            │                   │
│         │              │ Transformer  │            │                   │
│         │              │    ↓         │            │                   │
│         │  [噪声预测]    │    scheduler │            │                   │
│         └───────────────────┘         │            │                   │
│                    ↓                                   │
│         ┌───────────────────────────────┐                   │
│         │ 后处理 postprocess │               │
│         │  [去噪 actions] → [可执行动作] │              │
│         └───────────────────────────────┘               │
│                    ↓                                   │
│              保存 actions.pt  → [30,2,16]                   │
│              保存 latents.pt  → [1,48,F,H']                   │
│                    ↓                                   │
│         ┌───────────────────────────────┐                   │
│         │ KV Cache 更新 compute_kv_cache │                   │
│         │ [真实观察] → [B,144,F,H'] → 存入 Cache        │
│         │ [真实动作] → [B,L,3072] → 存入 Cache          │
│         └───────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 关键术语表

| 术语 | 含义 | 相关文件位置 |
|------|------|-------------|
| **Latent** | VAE 编码后的压缩表示 | `wan_va_server.py:369` |
| **Embedding** | 将输入映射到模型维度空间 | `model.py:830-842` |
| **Hidden State** | Transformer 内部的特征表示 | `model.py:860-866` |
| **KV Cache** | 缓存历史 Key/Value 加速注意力 | `model.py:396-409` |
| **Timestep** | 去噪过程中的时间步 | `wan_va_server.py:463-478` |
| **Noise Prediction** | Transformer 预测的噪声 | `model.py:497-501` |
| **CFG** | Classifier Free Guidance | `wan_va_server.py:508-511` |
| **Scheduler** | 去噪调度器（控制噪声去除） | `wan_va/utils/scheduler.py` |

---

## 调试和理解建议

### 如何追踪数据流

1. **添加打印语句**：
```python
# 在 _infer 中
logger.info(f"Input latent shape: {latents.shape}")
logger.info(f"Timestep: {t}")
logger.info(f"Output shape: {video_noise_pred.shape}")
```

2. **可视化中间结果**：
```python
# 检查 KV cache 内容
print(f"Cache key shape: {kv_cache['k'].shape}")
print(f"Cache mask: {kv_cache['mask']}")
```

3. **使用断言验证形状**：
```python
assert actions.shape[0] == self.job_config.action_dim
assert actions.shape[1] == frame_chunk_size
assert actions.shape[2] == self.action_per_frame
```

---

## 快速参考

### 文件索引

| 文件 | 主要功能 | 关键函数 |
|------|---------|----------|
| `eval_polict_client_openpi.py` | 评估主循环 | `eval_policy:440-608` |
| `wan_va_server.py` | 推理服务器 | `_reset:372`, `_infer:438`, `_compute_kv_cache:567` |
| `model.py` | Transformer 模型 | `forward:800`, `WanAttention:414` |
| `va_robotwin_cfg.py` | 配置文件 | 所有配置参数 |
| `scheduler.py` | 去噪调度器 | `step` 方法 |

### 配置参数影响

| 参数 | 增加值 | 影响 |
|------|--------|------|
| `num_inference_steps` | 更高 | 视频质量更高，推理更慢 |
| `action_num_inference_steps` | 更高 | 动作质量更高，推理更慢 |
| `attn_window` | 更大 | 可以看更远的历史，更多计算 |
| `guidance_scale` | 更高 | 更强的文本引导，但可能过饱和 |
| `action_guidance_scale` | 更高 | 更强的动作引导 |

---

## 常见问题

### Q1: 为什么有三个相机但视频不是三个视角的？

A1: 编码时三个视角被合并到统一的 latent 空间，但解码时只输出低位相机的视频（128×160）。高位相机的 latent 只作为条件信息参与推理。

### Q2: KV Cache 是如何工作的？

A2: KV Cache 存储历史观察和动作的 Key/Value，让当前 Token 能看到历史上下文，而无需重新计算。窗口大小由 `attn_window` 控制（默认 72）。

### Q3: actions.pt 和 latents.pt 的区别？

A3: `actions.pt` 是可执行的原始动作 `[30, 2, 16]`；`latents.pt` 是视频的潜在表示 `[1, 48, F, H']`，需要解码才能可视化。

### Q4: 如何添加噪音到 action expert？

A4: 修改 `wan_va_server.py:451-457` 中的初始化：
```python
noise_scale = getattr(self.job_config, 'action_noise_scale', 1.0)
actions = torch.randn(...) * noise_scale
```

---

## 完整执行流程

```
启动评估
    ↓
初始化环境
    ↓
循环 (直到任务完成)
    ↓
    ┌─────────────────────┐
    │ 重置模型         │
    └─────────────────────┘
    ↓
    ┌─────────────────────┐
    │ 推理生成动作     │
    └─────────────────────┘
    ↓
    ┌─────────────────────┐
    │ 执行动作序列     │
    └─────────────────────┘
    ↓
    ┌─────────────────────┐
    │ 更新 KV Cache     │
    └─────────────────────┘
    ↓
检查任务完成
    ↓
    保存可视化视频
    ↓
结束
```
