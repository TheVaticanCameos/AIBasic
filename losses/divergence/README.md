# KL Divergence

优先级：**P1**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

KL(P||Q)=mean_batch Σ_classes P*(log P-log Q)。

## Tensor 约定

log_p/log_q:[B,C]，要求有限且已归一化的 log probabilities；输出 []。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

指数得到 P 权重，计算 log-prob 差，类别求和、batch 求均值。

## 易错点与支持边界

KL 非对称。函数保留两侧梯度，蒸馏等场景由调用方 detach teacher。此实现不处理 log_p=-inf 的精确零概率。

## 复杂度

O(BC) 时间。

## 接口与依赖

`kl_divergence(log_p: Tensor, log_q: Tensor) -> Tensor`

log_p/log_q [B,C]，为归一化后的有限 log-prob -> scalar []。

若 P 是监督目标，调用方传 log_p.detach()；本函数保留两侧梯度。

仓库内依赖：无；可单独复制 implementation.py，仅依赖标准库/PyTorch。

## 运行与验证

在仓库根目录运行：

```bash
python -m losses.divergence.implementation
python -m pytest losses/divergence/test_module.py
```

测试关注：

- `test_kl_direction_reduction_and_gradient`

## 面试追问与练习

KL(P||Q) 和 KL(Q||P) 为什么不同？batchmean 与 mean 有何差异？

[独立手写题与起始接口](../../practice/prompts/losses__divergence.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
