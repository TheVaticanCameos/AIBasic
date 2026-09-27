# Adam 与 AdamW

优先级：**P0**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

m=β1*m+(1-β1)g；v=β2*v+(1-β2)g²；用 m/(1-β1^t)、v/(1-β2^t) 更新。

## Tensor 约定

每参数状态 first/second 与参数同形状 [...]；step 为每参数独立整数。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

维护两阶矩、偏差修正、sqrt(vhat)+eps；AdamW 在梯度更新外执行 θ*=1-lr*wd。

## 易错点与支持边界

Adam 的 L2 项加入梯度；AdamW 解耦衰减。grad=None 不递增该参数 step；不实现 AMSGrad/param groups/序列化。

## 复杂度

O(P) 时间，额外 O(2P) 矩状态。

## 接口与依赖

`zero_grad(self) -> None`

见实现中的类型与形状注释。

`step(self) -> None`

见实现中的类型与形状注释。

仓库内依赖：无；可单独复制 implementation.py，仅依赖标准库/PyTorch。

## 运行与验证

在仓库根目录运行：

```bash
python -m optimizers.adam.implementation
python -m pytest optimizers/adam/test_module.py
```

测试关注：

- `test_adam_matches_torch_across_steps_and_missing_gradients`

## 面试追问与练习

为什么需要偏差修正？Adam 加 L2 与 AdamW 为何不等价？

[独立手写题与起始接口](../../practice/prompts/optimizers__adam.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
