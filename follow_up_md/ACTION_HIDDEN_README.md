# 动作隐变量 (Action Hidden States) 接口使用说明

## 概述

这个功能允许你直接在隐空间中操作动作，而不是在原始动作空间。这样可以：

1. **拼接不同来源的动作**：将多个策略的动作在隐空间中拼接
2. **隐空间操作**：在隐空间中进行插值、加权等操作
3. **保存和加载隐状态**：保存预计算的隐变量以加速推理

## 修改的文件

- `wan_va/modules/model_with_action_hidden.py` - 修改后的模型文件，添加了动作隐变量接口
- `use_action_hidden.py` - 使用示例脚本

## 主要 API

### 1. `get_action_hidden_states(actions)`

将动作张量转换为隐空间表示。

**参数:**
- `actions` (torch.Tensor): 形状为 `[B, C, F, H, W]` 或 `[B, C, F, H, 1]` 的动作张量

**返回:**
- `torch.Tensor`: 形状为 `[B, L, C]` 的隐状态，其中 `L = F * H * W`

**示例:**
```python
from wan_va.modules.model_with_action_hidden import WanTransformer3DModel

model = WanTransformer3DModel(...)
actions = torch.randn(1, 30, 2, 16, 1).to(device)
hidden_states = model.get_action_hidden_states(actions)
# hidden_states.shape: [1, 32, 3072]
```

### 2. `concat_action_hidden_states(action_hidden_list)`

拼接多个动作隐状态。

**参数:**
- `action_hidden_list` (List[torch.Tensor]): 隐状态列表，每个形状为 `[B, L_i, C]`

**返回:**
- `torch.Tensor`: 拼接后的隐状态，形状为 `[B, sum(L_i), C]`

**示例:**
```python
hidden_1 = model.get_action_hidden_states(actions_1)  # [1, 32, 3072]
hidden_2 = model.get_action_hidden_states(actions_2)  # [1, 32, 3072]
concatenated = model.concat_action_hidden_states([hidden_1, hidden_2])
# concatenated.shape: [1, 64, 3072]
```

### 3. `forward(..., action_hidden_states=None)`

原有的 forward 方法，新增了 `action_hidden_states` 参数。

**新参数:**
- `action_hidden_states` (torch.Tensor, optional): 预计算的动作隐状态，形状为 `[B, L, C]`
  - 如果提供，跳过 `action_embedder` 直接使用这些隐状态
  - 如果为 `None`，使用原有的逻辑从 `noisy_latents` 计算隐状态

**示例:**
```python
output = model(
    input_dict=input_dict,
    action_mode=True,
    action_hidden_states=concatenated_hidden  # 直接使用预计算的隐状态
)
```

## 使用场景

### 场景 1: 拼接不同实验的动作

```python
# 从两个实验加载动作
actions_exp1 = load_actions('experiment_1/actions.pt')
actions_exp2 = load_actions('experiment_2/actions.pt')

# 转换为隐状态
hidden_exp1 = model.get_action_hidden_states(actions_exp1)
hidden_exp2 = model.get_action_hidden_states(actions_exp2)

# 拼接
concatenated = model.concat_action_hidden_states([hidden_exp1, hidden_exp2])

# 用于推理
output = model(input_dict=input_dict, action_mode=True, action_hidden_states=concatenated)
```

### 场景 2: 加权混合多个策略

```python
# 来自不同策略的动作
actions_policy_a = get_policy_a_actions()
actions_policy_b = get_policy_b_actions()

# 获取隐状态
hidden_a = model.get_action_hidden_states(actions_policy_a)
hidden_b = model.get_action_hidden_states(actions_policy_b)

# 加权组合 (70% 策略 A, 30% 策略 B)
weighted_a = hidden_a * 0.7
weighted_b = hidden_b * 0.3
concatenated = torch.cat([weighted_a, weighted_b], dim=1)

# 推理
output = model(input_dict=input_dict, action_mode=True, action_hidden_states=concatenated)
```

### 场景 3: 保存和加载隐状态

**保存隐状态:**
```python
# 在推理过程中保存
action_hidden = model.get_action_hidden_states(current_actions)
torch.save(action_hidden, 'saved_hidden_states.pt')
```

**加载隐状态:**
```python
# 在后续推理中加载
saved_hidden = torch.load('saved_hidden_states.pt', map_location=device)

# 可以直接使用或与其他隐状态拼接
output = model(
    input_dict=input_dict,
    action_mode=True,
    action_hidden_states=saved_hidden
)
```

### 场景 4: 隐空间插值

```python
# 在两个动作之间进行插值
hidden_start = model.get_action_hidden_states(actions_start)
hidden_end = model.get_action_hidden_states(actions_end)

# 线性插值
alpha = 0.5  # 50% 插值
interpolated = (1 - alpha) * hidden_start + alpha * hidden_end

# 推理
output = model(input_dict=input_dict, action_mode=True, action_hidden_states=interpolated)
```

## 如何替换原模型

在你的服务器或推理代码中，将导入从：

```python
from wan_va.modules.model import WanTransformer3DModel
```

改为：

```python
from wan_va.modules.model_with_action_hidden import WanTransformer3DModel
```

其他代码无需修改，新增的参数是可选的。

## 注意事项

1. **形状一致性**: 确保所有输入的动作隐状态具有相同的 `batch_size` 和 `inner_dim`（C 维度）

2. **设备一致性**: 确保所有张量在同一设备上（CPU 或 GPU）

3. **数据类型**: 模型通常使用 `torch.bfloat16`，确保输入数据类型匹配

4. **缓存更新**: 使用 `action_hidden_states` 时，注意 KV cache 的更新逻辑

## 示例运行

运行示例脚本：

```bash
python use_action_hidden.py
```

这将展示三种使用方式：
1. 基本拼接
2. 使用保存的隐状态
3. 加权拼接
