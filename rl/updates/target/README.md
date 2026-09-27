# 目标网络同步与参数冻结

优先级：**P0**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

hard: θtarget=θ；soft: θtarget=(1-τ)θtarget+τθ。

## Tensor 约定

源/目标 state_dict 键及张量形状一致；frozen_parameters 保持输入梯度。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

no_grad 下更新参数和 buffers；冻结上下文保存并恢复 requires_grad 标记。

## 易错点与支持边界

τ 表示新参数的权重；不要与另一种使用 ρ 表示保留比例的约定混淆。整数 buffers 直接复制。

## 复杂度

同步 O(P)，冻结遍历 O(参数张量数)。

## 接口与依赖

`hard_update(target: nn.Module, source: nn.Module) -> None`

复制参数和 buffers；目标与源必须同结构。

`soft_update(target: nn.Module, source: nn.Module, tau: float=0.005) -> None`

每个同形状参数 [...]：target <- (1-tau)*target + tau*source。

floating buffers 同样插值，整数 buffers 直接复制。

`frozen_parameters(*modules: nn.Module) -> Iterator[None]`

临时关闭参数梯度，但保留对网络输入（如 actor 动作）的梯度。

仓库内依赖：无；可单独复制 implementation.py，仅依赖标准库/PyTorch。

## 运行与验证

在仓库根目录运行：

```bash
python -m rl.updates.target.implementation
python -m pytest rl/updates/target/test_module.py
```

测试关注：

- `test_polyak_and_hard_copy`
- `test_freezing_preserves_input_gradient_and_restores_after_exception`

## 面试追问与练习

target network 为什么缓解不稳定？冻结参数与 no_grad 有什么区别？

[独立手写题与起始接口](../../../practice/prompts/rl__updates__target.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
