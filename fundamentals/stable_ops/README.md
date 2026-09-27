# 稳定 softmax / log-softmax / logsumexp

优先级：**P0**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

logsumexp(x)=m+log Σ exp(x-m)；log_softmax=x-logsumexp(x)。

## Tensor 约定

输入 floating [...,D,...]；沿 dim 归约，keepdim 决定是否保留单例轴。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

取最大值、平移、指数求和。logsumexp 恢复偏移；log-softmax 直接用平移后的 logits 减去归约项，避免巨大公共偏移的消减误差；softmax 由 log-softmax 指数化。

## 易错点与支持边界

先 softmax 再 log 会丢失小概率精度。此单元要求有限 logits；全 -inf mask 由 Attention 单元专门处理。

## 复杂度

O(N) 时间；归约临时张量随输入大小增长。

## 接口与依赖

`logsumexp(x: Tensor, dim: int=-1, keepdim: bool=False) -> Tensor`

x: floating [..., D, ...]；输出沿 dim 归约，要求输入为有限值。

`log_softmax(x: Tensor, dim: int=-1) -> Tensor`

x、返回值: 同形状 floating tensor；dim 是类别维度。

`softmax(x: Tensor, dim: int=-1) -> Tensor`

x、返回值: 同形状；沿 dim 的概率和为 1。

仓库内依赖：无；可单独复制 implementation.py，仅依赖标准库/PyTorch。

## 运行与验证

在仓库根目录运行：

```bash
python -m fundamentals.stable_ops.implementation
python -m pytest fundamentals/stable_ops/test_module.py
```

测试关注：

- `test_large_logits_and_gradients`
- `test_large_common_offset_preserves_uniform_probabilities`

## 面试追问与练习

为什么平移不改变 softmax？为什么交叉熵应直接从 logits 计算？

[独立手写题与起始接口](../../practice/prompts/fundamentals__stable_ops.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
