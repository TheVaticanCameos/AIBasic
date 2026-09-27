# SGD 与 Momentum

优先级：**P0**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

g=grad+λθ；v=μv+g；θ←θ-lr*v。

## Tensor 约定

parameter、gradient、velocity 均为各参数原形状 [...]。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

跳过 grad=None；可选 L2 项；更新动量；no_grad 下原地更新参数。

## 易错点与支持边界

zero_grad 与 step 职责分开。未实现 Nesterov、dampening、param groups、稀疏梯度和 checkpoint 序列化。

## 复杂度

O(P) 时间，Momentum 额外 O(P) 状态。

## 接口与依赖

`zero_grad(self) -> None`

见实现中的类型与形状注释。

`step(self) -> None`

见实现中的类型与形状注释。

仓库内依赖：无；可单独复制 implementation.py，仅依赖标准库/PyTorch。

## 运行与验证

在仓库根目录运行：

```bash
python -m optimizers.sgd.implementation
python -m pytest optimizers/sgd/test_module.py
```

测试关注：

- `test_multiple_steps_match_torch_with_decay_and_skipped_gradient`

## 面试追问与练习

动量如何减少震荡？grad=None 和全零梯度对 weight decay 的影响相同吗？

[独立手写题与起始接口](../../practice/prompts/optimizers__sgd.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
