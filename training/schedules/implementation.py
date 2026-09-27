"""纯函数学习率计划；调用方设置 optimizer.param_groups 的 lr。"""

import math


def warmup_cosine(step: int, warmup_steps: int, total_steps: int, min_ratio: float = 0.0) -> float:
    """step 从 0 开始；返回 base_lr 的倍率；warmup 最后一步达到 1。"""
    if step < 0 or not 0 <= warmup_steps < total_steps or not 0 <= min_ratio <= 1:
        raise ValueError("invalid schedule arguments")
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min((step - warmup_steps) / max(total_steps - warmup_steps - 1, 1), 1.0)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


if __name__ == "__main__":
    print([round(warmup_cosine(i, 2, 8), 3) for i in range(8)])
