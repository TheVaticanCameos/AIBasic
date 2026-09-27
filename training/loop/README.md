# 训练循环、梯度累积与裁剪

优先级：**P0**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

逻辑 batch loss=Σ_i (Bi/ΣBj)*mean_loss_i；全局 L2 norm=sqrt(Σ_p ||grad_p||²)。

## Tensor 约定

microbatch x:[Bi,...], y:[Bi,...]；损失 []；每个梯度与对应参数同形状。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

train、zero_grad、按样本数缩放各 microbatch loss、backward 累积、可选裁剪、统一 step。

## 易错点与支持边界

不同大小 microbatch 不能简单平均各 loss。大 batch 等价要求逐样本可分解损失；BatchNorm、Dropout、InfoNCE 等会破坏条件。

## 复杂度

计算量近似总 batch 前后向；峰值激活由最大 microbatch 决定。

## 接口与依赖

`clip_grad_norm(parameters: Iterable[Tensor], max_norm: float) -> Tensor`

梯度各为参数形状 [...]；返回裁剪前全局 L2 norm scalar []。

`train_step(model: nn.Module, microbatches: list[tuple[Tensor, Tensor]], optimizer: OptimizerLike, loss_fn: Callable[[Tensor, Tensor], Tensor], max_grad_norm: float | None=None) -> float`

每批 x [Bi,...], y [Bi,...]；loss_fn 返回样本平均损失 []。

各批按 Bi/总样本数加权。要求 loss 可按样本分解；BatchNorm、Dropout 等
会令 microbatch 与大 batch 的行为不同，不能据此宣称完全等价。

`zero_grad(self) -> None`

见实现中的类型与形状注释。

`step(self) -> object`

见实现中的类型与形状注释。

仓库内依赖：无；可单独复制 implementation.py，仅依赖标准库/PyTorch。

## 运行与验证

在仓库根目录运行：

```bash
python -m training.loop.implementation
python -m pytest training/loop/test_module.py
```

测试关注：

- `test_unequal_microbatches_match_full_batch_update`
- `test_global_clipping_matches_torch`

## 面试追问与练习

为什么 optimizer.step 不能放进累积循环？梯度裁剪应在什么时候执行？

[独立手写题与起始接口](../../practice/prompts/training__loop.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
