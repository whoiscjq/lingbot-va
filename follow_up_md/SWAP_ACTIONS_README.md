# 交换 actions.pt 文件以查看对 LLM 的影响

## 概述

通过加载和交换 `actions.pt` 文件，你可以：

1. **对比不同实验的动作输出** - 看不同配置如何影响动作
2. **研究动作变化对 LLM 的影响** - 分析动作序列如何影响模型
3. **排查动作序列问题** - 检查某些帧的动作是否异常
4. **快速实验不同策略** - 不需要重新运行完整推理

## 使用方法

### 命令格式

```bash
python swap_actions_from_pt.py <command> <arguments>
```

### 可用命令

#### 1. `analyze` - 分析动作统计

查看目录中所有动作文件的统计信息。

**用法:**
```bash
python swap_actions_from_pt.py analyze <directory> [pattern]
```

**示例:**
```bash
# 分析默认的 actions_*.pt 文件
python swap_actions_from_pt.py analyze train_out/real/

# 分析特定模式的文件
python swap_actions_from_pt.py analyze train_out/real/ "actions_0.pt"
```

**输出:**
- 每个文件的形状、范围、均值、标准差、范数
- 全局统计（形状是否一致、范数分布等）

---

#### 2. `swap` - 交换动作

用指定的动作文件替换推理中的动作。

**用法:**
```bash
python swap_actions_from_pt.py swap <base_dir> <source_action.pt> [output_dir]
```

**示例:**
```bash
# 用实验B的动作替换实验A的动作
python swap_actions_from_pt.py swap \
    train_out/experiment_A/ \
    train_out/experiment_B/actions_0.pt

# 指定输出目录
python swap_actions_from_pt.py swap \
    train_out/experiment_A/ \
    train_out/experiment_B/actions_0.pt \
    train_out/swapped/
```

**输出:**
- 加载源动作文件并显示统计信息
- 对比源动作和原始动作的差异
- 将交换后的动作保存到 `swapped_actions/` 目录（或指定的输出目录）

---

#### 3. `compare` - 对比动作序列

对比两个目录中的动作序列。

**用法:**
```bash
python swap_actions_from_pt.py compare <dir1> <dir2> [output_file]
```

**示例:**
```bash
# 对比两个实验
python swap_actions_from_pt.py compare \
    train_out/experiment_A/ \
    train_out/experiment_B/

# 指定输出文件
python swap_actions_from_pt.py compare \
    train_out/experiment_A/ \
    train_out/experiment_B/ \
    comparison_results.txt
```

**输出:**
- 逐个文件对比形状、最大差异、平均差异、L2 范数
- 保存详细对比结果到文本文件

---

## 使用场景

### 场景 1: 对比不同噪音强度的影响

```bash
# 分析不同噪音设置的推理结果
python swap_actions_from_pt.py analyze train_out/noise_1.0/
python swap_actions_from_pt.py analyze train_out/noise_2.0/
python swap_actions_from_pt.py analyze train_out/noise_0.5/

# 对比两个噪音设置
python swap_actions_from_pt.py compare \
    train_out/noise_1.0/ \
    train_out/noise_0.5/
```

### 场景 2: 研究特定动作帧

```bash
# 找到某个关键帧在所有实验中的动作
python swap_actions_from_pt.py analyze train_out/

# 如果发现某个帧异常，可以手动修改该帧的.pt文件
# 然后重新用修改后的动作进行推理
```

### 场景 3: 交换动作以观察 LLM 反应

```bash
# 实验1: 使用动作A推理
python wan_va_server.py --config-name robotwin
# 结果保存在 train_out/exp1/

# 实验2: 使用动作B推理
python wan_va_server.py --config-name robotwin
# 结果保存在 train_out/exp2/

# 分析对比
python swap_actions_from_pt.py compare \
    train_out/exp1/ \
    train_out/exp2/
```

### 场景 4: 快速验证动作序列

```bash
# 不需要重新运行完整推理，直接加载和检查动作
python swap_actions_from_pt.py analyze train_out/test_run/

# 如果发现问题，可以创建修正版本
# 然后交换到原始结果中验证
python swap_actions_from_pt.py swap \
    train_out/test_run/ \
    my_corrected_actions.pt
```

---

## actions.pt 文件格式

### 标准格式

```python
# 加载
actions = torch.load('actions_0.pt')

# 典型形状
# actions.shape = [C, F, H]
# 例如: [30, 2, 16]

# 维度含义
# C = action_dim (动作维度，如 30)
# F = frame_chunk_size (帧数，如 2)
# H = action_per_frame (每帧动作数，如 16)
```

### 数据范围

```python
# 归一化前的动作范围通常在 [-1, 1]
# 具体范围取决于：
# - 动作类型（关节角度、位置、夹爪等）
# - 归一化方法（quantiles, minmax 等）
```

---

## 对 LLM 的影响分析

### 通过动作变化推断

**1. 动作序列的平滑度**
- 平滑的动作序列通常表示 LLM 理解连贯
- 剧烈跳变可能表示不稳定的推理

**2. 动作的一致性**
- 相似条件下动作应该相似
- 大差异可能表示敏感输入

**3. 动作的范围**
- 合理的范围表明模型学习正确
- 超出范围可能表示预测错误

**4. 动作的多样性**
- 适度的多样性是好的（不同策略）
- 过度的多样性可能表示不稳定

### 对比方法

```python
# 示例：对比两个实验的动作
actions_exp1 = torch.load('exp1/actions_0.pt')
actions_exp2 = torch.load('exp2/actions_0.pt')

# 计算每帧的差异
for frame in range(actions_exp1.shape[1]):
    diff = actions_exp1[:, frame] - actions_exp2[:, frame]
    print(f"Frame {frame}: diff_norm = {np.linalg.norm(diff):.4f}")

# 统计分析
print(f"平均差异: {np.mean(diffs)}")
print(f"最大差异: {np.max(diffs)}")
```

---

## 注意事项

1. **形状匹配**: 交换动作时确保形状匹配
   - 不匹配的文件会被跳过并警告

2. **数据类型**: pt 文件可能是 float32 或 bfloat16
   - 脚本会自动处理并转为 numpy

3. **归一化**: 注意不同实验的归一化参数
   - 比较前确保归一化方式一致
   - 或者在交换后重新归一化

4. **时序顺序**: actions_*.pt 的数字通常表示时间顺序
   - actions_0.pt → 第一个 chunk
   - actions_1.pt → 第二个 chunk
   - 等等

---

## 输出说明

### analyze 命令输出

```
Loaded actions from actions_0.pt
  Shape: (30, 2, 16)
  Dtype: torch.bfloat16
  Min: -0.987654, Max: 0.998765
  Mean: 0.023456, Std: 0.345678

actions_0.pt:
  形状: (30, 2, 16)
  范围: [-0.9877, 0.9988]
  均值±标准差: 0.0235 ± 0.3457
  范数: 1.8923e+00
```

### swap 命令输出

```
从 source_actions.pt 加载动作
================================================================================
Loaded actions from source_actions.pt
  Shape: (30, 2, 16)
  ...

找到 10 个动作文件

--- 处理 actions_0.pt (1/10) ---
  差异统计:
    最大绝对差: 1.234567e-01
    平均绝对差: 3.456789e-02
    L2 范数: 2.3456789e+00
  ✓ 已保存到 swapped_actions/swapped_actions_0.pt

...
完成! 所有交换后的动作已保存到: swapped_actions/
```

### compare 命令输出

```
对比 dir1 和 dir2 中的动作序列
================================================================================

--- 对比 1: actions_0.pt vs actions_0.pt ---
  形状: (30, 2, 16)
  最大绝对差: 1.234567e-01
  平均绝对差: 3.456789e-02
  L2 范数: 2.3456789e+00

---

结果已保存到: action_comparison.txt
```

---

## 进阶用法

### 结合其他脚本

可以与 `compare_actions.py` 结合使用：

```bash
# 1. 对比原始动作
python compare_actions.py train_out/real/

# 2. 交换特定动作
python swap_actions_from_pt.py swap \
    train_out/real/ \
    my_custom_actions.pt

# 3. 再次对比交换后的结果
python compare_actions.py train_out/real/swapped_actions/
```

### 批量处理

```bash
# 对比多个实验
for i in {1..10}; do
    python swap_actions_from_pt.py compare \
        train_out/exp$i/ \
        train_out/exp0/ \
        "comparison_exp$i.txt"
done
```

---

## 总结

通过交换 `actions.pt` 文件，你可以：

1. **快速对比** - 不需要重新运行推理
2. **隔离变量** - 保持其他条件不变，只改变动作
3. **深入分析** - 研究动作变化对模型的影响
4. **加速迭代** - 快速验证修改的效果

这确实是研究动作对 LLM 影响的有效方法！
