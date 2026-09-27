# Warmup + Cosine 学习率计划

优先级：**P1**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

warmup ratio=(step+1)/warmup_steps；之后 ratio=min+(1-min)*(1+cos(π*progress))/2。

## Tensor 约定

纯标量输入与输出；step 为从 0 开始的优化器更新计数。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

先 warmup、再按剩余区间余弦衰减；超过 total_steps 固定最小倍率。

## 易错点与支持边界

在 optimizer 更新级别计步；梯度累积时不要按 microbatch 计步。无 warmup 时首步倍率为 1。

## 复杂度

O(1) 时间与状态。

## 接口与依赖

`warmup_cosine(step: int, warmup_steps: int, total_steps: int, min_ratio: float=0.0) -> float`

step 从 0 开始；返回 base_lr 的倍率；warmup 最后一步达到 1。

仓库内依赖：无；可单独复制 implementation.py，仅依赖标准库/PyTorch。

## 运行与验证

在仓库根目录运行：

```bash
python -m training.schedules.implementation
python -m pytest training/schedules/test_module.py
```

测试关注：

- `test_schedule_boundaries`

## 面试追问与练习

warmup 如何影响初始更新？恢复训练时应恢复哪个计数？

[独立手写题与起始接口](../../practice/prompts/training__schedules.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
