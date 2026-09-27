# 交叉熵、Label Smoothing、BCE with Logits

优先级：**P0**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

CE=-log p_y；平滑目标为 (1-eps)*one_hot+eps/C；BCE=log(1+exp(z))-y*z。

## Tensor 约定

CE logits:[B,C]、target:int64 [B]；BCE logits/target 同形状 [...]；返回 scalar []。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

复用稳定 log-softmax；gather 目标类别；平滑混合均匀分布损失；BCE 用 logaddexp。

## 易错点与支持边界

CE 输入是 logits，不能先 softmax。BCE 在 z=0 的梯度也需正确；只支持 mean reduction，无 ignore_index/class weights。

## 复杂度

CE O(BC)，BCE O(N)。

## 接口与依赖

`cross_entropy(logits: Tensor, targets: Tensor, label_smoothing: float=0.0) -> Tensor`

logits [B,C], targets int64 [B] -> scalar []。

smoothing 使用 (1-eps)*one_hot + eps/C，与 PyTorch 定义一致。

`binary_cross_entropy_with_logits(logits: Tensor, targets: Tensor) -> Tensor`

logits/targets 同形状 [...]，targets 为 [0,1] 浮点值 -> scalar []。

仓库内依赖：[fundamentals.stable_ops.implementation](../../fundamentals/stable_ops/implementation.py)

## 运行与验证

在仓库根目录运行：

```bash
python -m losses.classification.implementation
python -m pytest losses/classification/test_module.py
```

测试关注：

- `test_cross_entropy_smoothing_and_extreme_logits`
- `test_bce_values_and_gradient_at_zero`

## 面试追问与练习

多标签分类为什么用 BCE？label smoothing 改变了什么目标？

[独立手写题与起始接口](../../practice/prompts/losses__classification.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
