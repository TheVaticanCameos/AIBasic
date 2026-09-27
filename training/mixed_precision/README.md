# AMP 混合精度训练

优先级：**P1**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

autocast 选择算子精度；scaler 放大 loss 后 backward，unscale 后裁剪，再 step/update。

## Tensor 约定

x:[B,...], y:[B,...]；loss scalar []；梯度形状与参数一致。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

在循环外创建 scaler；前向进入 autocast；缩放 loss、反向、恢复梯度尺度、裁剪、更新。

## 易错点与支持边界

CPU 用 bfloat16 和 disabled scaler；CUDA 用 float16 与 enabled scaler。不能在 unscale 之前裁剪；本地验证仅覆盖 CPU。

## 复杂度

计算量同常规训练，部分激活精度降低；GPU 的吞吐收益依硬件/算子而定。

## 接口与依赖

`amp_train_step(model: nn.Module, x: Tensor, y: Tensor, optimizer: torch.optim.Optimizer, loss_fn: Callable[[Tensor, Tensor], Tensor], scaler: torch.amp.GradScaler, enabled: bool=True, max_grad_norm: float | None=None) -> float`

x [B,...], y [B,...] -> loss；model/x/y 必须在同一设备。

CPU 用 bfloat16 + disabled scaler；CUDA 用 float16 + enabled scaler。
scaler 应在训练循环外创建并跨 step 复用，以保留动态缩放状态。

仓库内依赖：无；可单独复制 implementation.py，仅依赖标准库/PyTorch。

## 运行与验证

在仓库根目录运行：

```bash
python -m training.mixed_precision.implementation
python -m pytest training/mixed_precision/test_module.py
```

测试关注：

- `test_cpu_amp_updates_parameters`

## 面试追问与练习

为什么 float16 通常需要 loss scaling？为什么 autocast 不包住 backward？

[独立手写题与起始接口](../../practice/prompts/training__mixed_precision.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
