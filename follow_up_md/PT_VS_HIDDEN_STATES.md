# 原代码保存的 .pt 文件 vs 动作隐状态

## 结论：**不是同一个东西**

### 1. 原代码保存的 actions.pt 文件

在 `wan_va_server.py:560-561` 中保存：

```python
save_async(latents, os.path.join(self.exp_save_root, f'latents_{frame_st_id}.pt'))
save_async(actions, os.path.join(self.exp_save_root, f'actions_{frame_st_id}.pt'))
```

**特点：**
- **内容**：去噪后的**最终动作**
- **形状**：`[1, action_dim, frame_chunk_size, action_per_frame, 1]`
  - 例如：`[1, 30, 2, 16, 1]`
  - 这是**原始动作空间**的表示
- **位置**：在推理循环结束，经过 `action_proj_out` 和 `postprocess_action` 后保存
- **用途**：用于后续执行或分析
- **语义**："这是模型预测的最终动作序列"

**数据流：**
```
随机噪声 → [去噪循环] → action_proj_out → postprocess_action → actions.pt
          (transformer)    (隐→原始)     (反归一化)
```

---

### 2. 动作隐状态 (Action Hidden States)

通过 `get_action_hidden_states()` 获取：

```python
hidden_states = model.get_action_hidden_states(actions)
```

**特点：**
- **内容**：动作的**隐空间表示**
- **形状**：`[B, L, C]` 其中 `L = F * H * W`
  - 例如：`[1, 32, 3072]` (当 actions=[1,30,2,16,1])
  - `L = 2 * 16 = 32`
  - `C = inner_dim = 3072` (24 heads * 128 dim)
- **位置**：在 `action_embedder` 之后，transformer 之前
- **用途**：用于在隐空间中操作、拼接动作
- **语义**："这是动作在模型内部的表示"

**数据流：**
```
原始动作 → action_embedder → 动作隐状态 → [transformer] → action_proj_out → 最终动作
           (原始→隐)                           (隐→原始)
```

---

## 对比总结

| 特性 | actions.pt | 动作隐状态 |
|------|------------|-------------|
| **数据类型** | 原始动作值 | 隐空间向量 |
| **维度** | [1, 30, 2, 16, 1] | [1, 32, 3072] |
| **语义** | 可执行的动作 | 动作的内部表示 |
| **可读性** | 高（直接的关节角度等） | 低（抽象表示） |
| **可操作性** | 直接修改或执行 | 可在隐空间插值、拼接 |
| **处理阶段** | 去噪后 | 去噪前 |
| **是否经过模型** | 是（完整去噪） | 否（仅编码） |

---

## 图解关系

```
原始动作空间                      隐空间 (Hidden Space)
    |                                 |
    v                                 v
actions.pt ←───────────── action_embedder ←─────────→ 动作隐状态
[1,30,2,16,1]                    [1,32,3072]
    |                                 |
    | (postprocess)                   |
    v                                 v
最终可执行动作 ←───────────── action_proj_out ←───────────> Transformer 处理
                                    [1,32,3072]
```

---

## 实际使用场景

### 使用 actions.pt（原始动作）

```python
# 加载保存的动作
actions = torch.load('actions_0.pt')  # [1, 30, 2, 16, 1]

# 直接用于环境执行
env.take_action(actions[0, :, 0, :, 0])  # 第0帧的动作
env.take_action(actions[0, :, 1, :, 0])  # 第1帧的动作

# 或者分析动作分布
analyze_action_distribution(actions)
```

### 使用动作隐状态

```python
# 将动作转换为隐状态
actions = torch.load('actions_0.pt')
hidden_states = model.get_action_hidden_states(actions)  # [1, 32, 3072]

# 在隐空间操作
hidden_states_a = model.get_action_hidden_states(actions_a)
hidden_states_b = model.get_action_hidden_states(actions_b)

# 拼接隐状态
concatenated = torch.cat([hidden_states_a, hidden_states_b], dim=1)

# 用于模型推理
output = model(
    input_dict=input_dict,
    action_mode=True,
    action_hidden_states=concatenated
)
```

---

## 转换关系

### 从 actions.pt → 动作隐状态

```python
actions = torch.load('actions_0.pt')  # [1, 30, 2, 16, 1]
hidden_states = model.get_action_hidden_states(actions)
# 结果：[1, 32, 3072]
```

### 从动作隐状态 → actions.pt

**无法直接转换！** 因为：
- 隐状态 → 需要 Transformer 和去噪 → 最终动作
- 这个过程依赖文本条件、时间步等

但可以重新去噪：
```python
hidden_states = torch.load('saved_hidden_states.pt')

# 在 forward 中使用
output = model(
    input_dict={
        'noisy_latents': torch.randn(...),  # 需要噪声
        'timesteps': torch.tensor([100]),
        'grid_id': grid_id,
        'text_emb': text_embed,
    },
    action_mode=True,
    action_hidden_states=hidden_states
)
# output 可以通过后续步骤转换为可执行动作
```

---

## 总结

- **actions.pt** = 去噪完成后的**最终结果**，可以直接使用
- **动作隐状态** = 动作在模型内部的**中间表示**，用于模型内部的拼接和操作

如果你想要：
1. **执行动作** → 使用 actions.pt
2. **拼接不同来源的动作** → 使用动作隐状态
3. **分析最终输出** → 使用 actions.pt
4. **在隐空间探索** → 使用动作隐状态
