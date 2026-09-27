# Max / Average Pooling

优先级：**P0**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

对每个 K×K 窗口取最大值或均值。

## Tensor 约定

[B,C,H,W] → windows:[B,C,Hout,Wout,K,K] → [B,C,Hout,Wout]。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

unfold 创建窗口视图；flatten 最后两个轴后 max，或对两个轴 mean。

## 易错点与支持边界

限定无 padding、floor 模式；窗口尺寸必须有效。max 仅向选中位置传梯度，重复最大值使用首个索引。

## 复杂度

O(B*C*Hout*Wout*K²) 时间；不含可训练参数。

## 接口与依赖

`pool2d(x: Tensor, kernel_size: int=2, stride: int | None=None, mode: Literal['max', 'avg']='max') -> Tensor`

x [B,C,H,W] -> [B,C,Hout,Wout]。

仓库内依赖：无；可单独复制 implementation.py，仅依赖标准库/PyTorch。

## 运行与验证

在仓库根目录运行：

```bash
python -m layers.pooling.implementation
python -m pytest layers/pooling/test_module.py
```

测试关注：

- `test_pooling_output_and_gradients`

## 面试追问与练习

MaxPool 与 AvgPool 的反向传播有什么区别？

[独立手写题与起始接口](../../practice/prompts/layers__pooling.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
