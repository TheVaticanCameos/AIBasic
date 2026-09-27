# PPO Clipped Objective

优先级：**P0**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

loss=-mean(min(r*A,clip(r,1-eps,1+eps)*A))；r=exp(logπ_new-logπ_old)。

## Tensor 约定

new/old log_probs/advantages:[B] → scalar loss 与标量诊断值。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

先算 log-ratio 再 exp；计算原目标与 clipped 目标；逐元素 min 后归约。

## 易错点与支持边界

负优势时也必须 min，不能先随意 clamp loss。old_log_probs 与 advantages 必须 detach。approx_kl 是采样估计，不是精确 KL。

## 复杂度

O(B) 时间与中间张量。

## 接口与依赖

`clipped_policy_loss(new_log_probs: Tensor, old_log_probs: Tensor, advantages: Tensor, clip_ratio: float=0.2) -> tuple[Tensor, PPOStats]`

输入均 [B] -> (loss [], stats)；仅 new_log_probs 接收梯度。

仓库内依赖：无；可单独复制 implementation.py，仅依赖标准库/PyTorch。

## 运行与验证

在仓库根目录运行：

```bash
python -m rl.objectives.ppo.implementation
python -m pytest rl/objectives/ppo/test_module.py
```

测试关注：

- `test_positive_negative_advantage_clipping_and_gradient_boundaries`
- `test_unfavorable_ratios_are_not_clipped_away`

## 面试追问与练习

为什么 PPO clipping 不能严格保证 KL 上界？正负优势各在哪侧停止梯度？

[独立手写题与起始接口](../../../practice/prompts/rl__objectives__ppo.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
