# Generalized Advantage Estimation

优先级：**P0**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

δ_t=r_t+γ*(1-terminated_t)*V(next)-V_t；A_t=δ_t+γλ*(1-ended_t)*A_(t+1)。

## Tensor 约定

输入均 [T,B]；输出 (advantages,returns)，各 [T,B]，returns=A+V。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

逆序累加 TD residual；bootstrap mask 只看 terminated，trace mask 看 terminated 或 truncated。

## 易错点与支持边界

rollout 尾部 trace 为零但 delta 仍 bootstrap；不允许跨 reset 串联。所有结果 detach。

## 复杂度

O(TB) 时间与输出内存。

## 接口与依赖

`generalized_advantage_estimate(rewards: Tensor, values: Tensor, next_values: Tensor, terminated: Tensor, truncated: Tensor, gamma: float=0.99, gae_lambda: float=0.95) -> tuple[Tensor, Tensor]`

所有输入 [T,B] -> (advantages, returns)，各 [T,B]，均无梯度。

next_values 使用 transition 的下一观测；episode reset 不得覆盖它。
δ=r+γ(1-terminated)V(next)-V；A=δ+γλ(1-ended)A_next。

仓库内依赖：无；可单独复制 implementation.py，仅依赖标准库/PyTorch。

## 运行与验证

在仓库根目录运行：

```bash
python -m rl.returns.gae.implementation
python -m pytest rl/returns/gae/test_module.py
```

测试关注：

- `test_hand_computed_gae_with_episode_boundary`

## 面试追问与练习

λ=0 与 λ=1 分别对应什么？为什么需要两种 mask？

[独立手写题与起始接口](../../../practice/prompts/rl__returns__gae.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
