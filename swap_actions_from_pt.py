"""
从保存的 actions.pt 文件加载动作并用于推理
可以用来：
1. 对比不同实验的动作输出
2. 研究动作变化对 LLM 的影响
3. 排查动作序列问题
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple


def load_actions_from_pt(pt_path: str) -> np.ndarray:
    """
    从 pt 文件加载动作

    Args:
        pt_path: pt 文件路径

    Returns:
        actions: numpy array, shape [C, F, H]
    """
    actions_tensor = torch.load(pt_path, map_location='cpu', weights_only=False)
    print(f"Loaded actions from {pt_path}")
    print(f"  Shape: {actions_tensor.shape}")
    print(f"  Dtype: {actions_tensor.dtype}")
    print(f"  Min: {actions_tensor.min():.6f}, Max: {actions_tensor.max():.6f}")
    print(f"  Mean: {actions_tensor.mean():.6f}, Std: {actions_tensor.std():.6f}")
    return actions_tensor.numpy()


def load_all_actions_from_directory(directory: str, pattern: str = "actions_*.pt") -> List[Tuple[str, np.ndarray]]:
    """
    从目录加载所有匹配的动作文件

    Args:
        directory: 目录路径
        pattern: 文件匹配模式

    Returns:
        List of (filename, actions) tuples
    """
    dir_path = Path(directory)
    action_files = list(dir_path.glob(pattern))

    if not action_files:
        print(f"没有找到匹配 {pattern} 的文件在 {directory}")
        return []

    results = []
    for pt_file in sorted(action_files):
        try:
            actions = load_actions_from_pt(str(pt_file))
            results.append((pt_file.name, actions))
        except Exception as e:
            print(f"加载 {pt_file.name} 失败: {e}")

    return results


def swap_actions_in_inference(
    base_dir: str,
    source_action_file: str,
    output_dir: Optional[str] = None,
):
    """
    用保存的动作文件替换推理中的动作

    Args:
        base_dir: 原始推理输出目录
        source_action_file: 要使用的动作文件路径
        output_dir: 交换后动作的保存目录
    """
    print("=" * 80)
    print(f"从 {source_action_file} 加载动作")
    print("=" * 80)

    # 加载源动作
    source_actions = load_actions_from_pt(source_action_file)
    source_shape = source_actions.shape

    # 查找原始推理中的动作文件
    dir_path = Path(base_dir)
    action_files = list(dir_path.glob("actions_*.pt"))

    if not action_files:
        print(f"在 {base_dir} 中没有找到 actions_*.pt 文件")
        return

    print(f"\n找到 {len(action_files)} 个动作文件")

    # 创建输出目录
    if output_dir is None:
        output_dir = dir_path / "swapped_actions"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 对每个原始动作文件进行替换
    for i, pt_file in enumerate(sorted(action_files)):
        print(f"\n--- 处理 {pt_file.name} ({i+1}/{len(action_files)}) ---")

        # 加载原始动作（用于对比）
        original_actions = load_actions_from_pt(str(pt_file))
        original_shape = original_actions.shape

        # 检查形状匹配
        if source_shape != original_shape:
            print(f"  ⚠️  形状不匹配!")
            print(f"     源: {source_shape}")
            print(f"     原始: {original_shape}")
            print(f"     跳过此文件")
            continue

        # 计算差异
        diff = source_actions - original_actions
        abs_diff = np.abs(diff)
        print(f"  差异统计:")
        print(f"    最大绝对差: {abs_diff.max():.6f}")
        print(f"    平均绝对差: {abs_diff.mean():.6f}")
        print(f"    L2 范数: {np.linalg.norm(diff):.6f}")

        # 保存交换后的动作
        output_file = output_dir / f"swapped_{pt_file.name}"
        torch.save(torch.from_numpy(source_actions), output_file)
        print(f"  ✓ 已保存到 {output_file}")

    print(f"\n{'=' * 80}")
    print(f"完成! 所有交换后的动作已保存到: {output_dir}")
    print(f"{'=' * 80}")


def compare_action_sequences(dir1: str, dir2: str, output_file: str = "action_comparison.txt"):
    """
    对比两个目录中的动作序列

    Args:
        dir1: 第一个目录
        dir2: 第二个目录
        output_file: 对比结果输出文件
    """
    print("=" * 80)
    print(f"对比 {dir1} 和 {dir2} 中的动作序列")
    print("=" * 80)

    actions1 = load_all_actions_from_directory(dir1)
    actions2 = load_all_actions_from_directory(dir2)

    if len(actions1) != len(actions2):
        print(f"⚠️  文件数量不匹配: {len(actions1)} vs {len(actions2)}")

    results = []
    for i, ((name1, acts1), (name2, acts2)) in enumerate(zip(actions1, actions2)):
        print(f"\n--- 对比 {i+1}: {name1} vs {name2} ---")

        if acts1.shape != acts2.shape:
            print(f"  ⚠️  形状不匹配: {acts1.shape} vs {acts2.shape}")
            continue

        # 计算差异
        diff = acts1 - acts2
        abs_diff = np.abs(diff)

        stats = {
            'file1': name1,
            'file2': name2,
            'shape': acts1.shape,
            'max_abs_diff': float(abs_diff.max()),
            'mean_abs_diff': float(abs_diff.mean()),
            'std_abs_diff': float(abs_diff.std()),
            'l2_norm': float(np.linalg.norm(diff)),
        }

        print(f"  形状: {stats['shape']}")
        print(f"  最大绝对差: {stats['max_abs_diff']:.6e}")
        print(f"  平均绝对差: {stats['mean_abs_diff']:.6e}")
        print(f"  L2 范数: {stats['l2_norm']:.6e}")

        results.append(stats)

    # 保存结果
    with open(output_file, 'w') as f:
        f.write("动作序列对比结果\n")
        f.write("=" * 80 + "\n\n")
        for stats in results:
            f.write(f"文件1: {stats['file1']}\n")
            f.write(f"文件2: {stats['file2']}\n")
            f.write(f"形状: {stats['shape']}\n")
            f.write(f"最大绝对差: {stats['max_abs_diff']:.6e}\n")
            f.write(f"平均绝对差: {stats['mean_abs_diff']:.6e}\n")
            f.write(f"L2 范数: {stats['l2_norm']:.6e}\n")
            f.write("-" * 40 + "\n\n")

    print(f"\n结果已保存到: {output_file}")


def analyze_action_stats(directory: str, pattern: str = "actions_*.pt"):
    """
    分析目录中所有动作文件的统计信息

    Args:
        directory: 目录路径
        pattern: 文件匹配模式
    """
    print("=" * 80)
    print(f"分析 {directory} 中的动作文件")
    print("=" * 80)

    actions_list = load_all_actions_from_directory(directory, pattern)

    if not actions_list:
        print("没有找到动作文件")
        return

    print(f"\n找到 {len(actions_list)} 个动作文件\n")

    all_stats = []

    for filename, actions in actions_list:
        stats = {
            'filename': filename,
            'shape': actions.shape,
            'min': float(actions.min()),
            'max': float(actions.max()),
            'mean': float(actions.mean()),
            'std': float(actions.std()),
            'norm': float(np.linalg.norm(actions)),
        }

        all_stats.append(stats)

        print(f"{filename}:")
        print(f"  形状: {stats['shape']}")
        print(f"  范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  均值±标准差: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  范数: {stats['norm']:.4e}")

    # 全局统计
    print(f"\n{'=' * 80}")
    print("全局统计:")
    print(f"{'=' * 80}")

    shapes = [s['shape'] for s in all_stats]
    if len(set(shapes)) == 1:
        print(f"✓ 所有文件形状一致: {shapes[0]}")
    else:
        print(f"✗ 文件形状不一致:")
        for shape in set(shapes):
            count = sum(1 for s in shapes if s == shape)
            print(f"    {shape}: {count} 个文件")

    norms = [s['norm'] for s in all_stats]
    print(f"\n范数统计:")
    print(f"  最小: {min(norms):.4e}")
    print(f"  最大: {max(norms):.4e}")
    print(f"  平均: {np.mean(norms):.4e}")
    print(f"  标准差: {np.std(norms):.4e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法:")
        print("  1. 分析动作统计:")
        print(f"     python {sys.argv[0]} analyze <directory>")
        print("\n  2. 交换动作:")
        print(f"     python {sys.argv[0]} swap <base_dir> <source_action.pt> [output_dir]")
        print("\n  3. 对比动作序列:")
        print(f"     python {sys.argv[0]} compare <dir1> <dir2> [output_file]")
        sys.exit(1)

    command = sys.argv[1]
    args = sys.argv[2:]

    if command == "analyze":
        if len(args) < 1:
            print("错误: 需要指定目录")
            sys.exit(1)
        directory = args[0]
        pattern = args[1] if len(args) > 1 else "actions_*.pt"
        analyze_action_stats(directory, pattern)

    elif command == "swap":
        if len(args) < 2:
            print("错误: 需要指定 base_dir 和 source_action.pt")
            sys.exit(1)
        base_dir = args[0]
        source_file = args[1]
        output_dir = args[2] if len(args) > 2 else None
        swap_actions_in_inference(base_dir, source_file, output_dir)

    elif command == "compare":
        if len(args) < 2:
            print("错误: 需要指定两个目录")
            sys.exit(1)
        dir1 = args[0]
        dir2 = args[1]
        output_file = args[2] if len(actions) > 2 else "action_comparison.txt"
        compare_action_sequences(dir1, dir2, output_file)

    else:
        print(f"未知命令: {command}")
        print("可用命令: analyze, swap, compare")
        sys.exit(1)
