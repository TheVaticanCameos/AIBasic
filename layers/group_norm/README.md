# GroupNorm

优先级：**P1**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

将 C 分为 G 组，在每个样本的每组内计算均值和方差。

## Tensor 约定

[B,C,...] → [B,G,C/G*spatial] → 原形状。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

重排为组、末维归约、还原形状、逐通道 affine。

## 易错点与支持边界

C 必须被 G 整除；不跨 batch 归约，不需要 running statistics。

## 复杂度

O(N) 时间，参数 O(C)。

## 接口与依赖

`forward(self, x: Tensor) -> Tensor`

x [B,C,...] -> 同形状。

仓库内依赖：无；可单独复制 implementation.py，仅依赖标准库/PyTorch。

## 运行与验证

在仓库根目录运行：

```bash
python -m layers.group_norm.implementation
python -m pytest layers/group_norm/test_module.py
```

测试关注：

- `test_matches_torch_and_independent_of_other_samples`

## 面试追问与练习

G=1、G=C 分别接近哪种归一化？与 LayerNorm 的 affine 参数范围相同吗？

[独立手写题与起始接口](../../practice/prompts/layers__group_norm.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
