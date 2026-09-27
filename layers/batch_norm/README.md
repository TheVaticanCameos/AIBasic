# BatchNorm

优先级：**P0**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

训练使用批均值与总体方差；running <- (1-m)*running + m*batch_stat。

## Tensor 约定

x:[B,C,...]；统计量:[C]；广播形状:[1,C,1,...]。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

在 B 与空间轴归约；前向用有偏方差，running_var 更新用无偏方差；eval 使用 running statistics。

## 易错点与支持边界

momentum 是新统计量权重，与优化器动量语义不同。running 统计是 buffer，不参与梯度。每通道只有一个样本时拒绝训练。

## 复杂度

O(N) 时间，参数及 running buffers O(C)。

## 接口与依赖

`forward(self, x: Tensor) -> Tensor`

x [B,C,...] -> 同形状；每个 C 在 batch 与空间维度上归约。

仓库内依赖：无；可单独复制 implementation.py，仅依赖标准库/PyTorch。

## 运行与验证

在仓库根目录运行：

```bash
python -m layers.batch_norm.implementation
python -m pytest layers/batch_norm/test_module.py
```

测试关注：

- `test_train_eval_running_statistics_and_gradients`

## 面试追问与练习

为什么训练与 running 方差估计不同？batch 很小时会有什么问题？

[独立手写题与起始接口](../../practice/prompts/layers__batch_norm.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
