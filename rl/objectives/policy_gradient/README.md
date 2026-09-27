# Policy Gradient Loss

优先级：**P0**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

loss=-mean(log π(a|s)*weight)-entropy_coef*mean(entropy)。

## Tensor 约定

log_probs/weights/entropy:[B] → loss:[]。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

对 return 或 advantage 停止梯度，与当前策略 log-prob 相乘取负均值。

## 易错点与支持边界

正权重应提高该动作概率，负权重降低；权重通常由旧数据或 critic 产生，不能反向优化它。

## 复杂度

O(B) 时间。

## 接口与依赖

`policy_gradient_loss(log_probs: Tensor, weights: Tensor, entropy: Tensor | None=None, entropy_coef: float=0.0) -> Tensor`

log_probs/weights/entropy [B] -> scalar []；weights 明确停止梯度。

仓库内依赖：无；可单独复制 implementation.py，仅依赖标准库/PyTorch。

## 运行与验证

在仓库根目录运行：

```bash
python -m rl.objectives.policy_gradient.implementation
python -m pytest rl/objectives/policy_gradient/test_module.py
```

测试关注：

- `test_loss_sign_and_detached_weights`

## 面试追问与练习

baseline 为什么可以降低方差？为什么 detach advantage？

[独立手写题与起始接口](../../../practice/prompts/rl__objectives__policy_gradient.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
