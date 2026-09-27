# Inverted Dropout

优先级：**P0**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

训练 y=x*m/(1-p)，m~Bernoulli(1-p)；推理 y=x。

## Tensor 约定

输入、输出与随机 mask 均为 [...]；mask 为 bool。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

训练时独立采样保留 mask 并缩放；eval 时恒等；p=1 单独处理。

## 易错点与支持边界

p 表示丢弃率。训练时缩放保证期望不变；p=1 不能除以零。随机测试检查统计性质而非逐元素匹配不同实现的 RNG。

## 复杂度

O(N) 时间、O(N) mask。

## 接口与依赖

`forward(self, x: Tensor) -> Tensor`

x [...] -> 同形状；每个元素独立保留。

仓库内依赖：无；可单独复制 implementation.py，仅依赖标准库/PyTorch。

## 运行与验证

在仓库根目录运行：

```bash
python -m layers.dropout.implementation
python -m pytest layers/dropout/test_module.py
```

测试关注：

- `test_scaling_eval_and_edge_probabilities`

## 面试追问与练习

训练和推理为何行为不同？Dropout 与梯度累积一起使用会怎样？

[独立手写题与起始接口](../../practice/prompts/layers__dropout.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
