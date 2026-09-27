# Embedding 查表

优先级：**P1**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

output=weight[indices]；重复索引的梯度相加。

## Tensor 约定

indices:int64 [...]；weight:[V,D] → output:[...,D]。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

为词表建立可学习矩阵；利用索引选行；用重复 token 验证梯度累积。

## 易错点与支持边界

不实现 padding_idx、稀疏梯度和越界兼容行为；索引必须在 [0,V)。

## 复杂度

查表 O(ND)，参数 O(VD)；本实现 autograd 使用稠密权重梯度。

## 接口与依赖

`forward(self, indices: Tensor) -> Tensor`

indices int64 [...] -> floating [...,D]；不实现 padding_idx/sparse。

仓库内依赖：无；可单独复制 implementation.py，仅依赖标准库/PyTorch。

## 运行与验证

在仓库根目录运行：

```bash
python -m layers.embedding.implementation
python -m pytest layers/embedding/test_module.py
```

测试关注：

- `test_repeated_indices_accumulate_gradients`

## 面试追问与练习

Embedding 与 one-hot 乘 Linear 有何关系？重复 token 的梯度如何合并？

[独立手写题与起始接口](../../practice/prompts/layers__embedding.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
