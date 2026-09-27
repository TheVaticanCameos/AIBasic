# MSE 与 Huber

优先级：**P0**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

MSE=mean((pred-target)²)；Huber 小误差为 0.5e²，大误差为 δ*(|e|-0.5δ)。

## Tensor 约定

prediction/target 必须同形状 [...]；输出 scalar []。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

先检查形状；计算误差；Huber 按阈值选择二次或线性分支。

## 易错点与支持边界

[B,1] 与 [B] 相减会广播为 [B,B]，所以主动拒绝。Huber 与 SmoothL1 相差 delta 缩放。

## 复杂度

O(N) 时间。

## 接口与依赖

`mse(prediction: Tensor, target: Tensor) -> Tensor`

prediction/target 同形状 [...] -> scalar []；所有元素取均值。

`huber(prediction: Tensor, target: Tensor, delta: float=1.0) -> Tensor`

同形状 [...] -> scalar []；delta>0，区别于缩放后的 SmoothL1。

仓库内依赖：无；可单独复制 implementation.py，仅依赖标准库/PyTorch。

## 运行与验证

在仓库根目录运行：

```bash
python -m losses.regression.implementation
python -m pytest losses/regression/test_module.py
```

测试关注：

- `test_losses_match_and_reject_broadcasting`

## 面试追问与练习

Huber 为什么对异常值更稳健？MSE 的 mean 究竟归约哪些维度？

[独立手写题与起始接口](../../practice/prompts/losses__regression.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
