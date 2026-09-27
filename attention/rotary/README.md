# RoPE 旋转位置编码

优先级：**P1**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

每对维度施加二维旋转：[x_even*cosθ-x_odd*sinθ, x_even*sinθ+x_odd*cosθ]。

## Tensor 约定

x:[...,T,d]，angles:[T,d/2]，要求 d 为偶数。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

相邻偶奇维配对、生成角度、旋转、交错还原；通过 offset 支持增量位置。

## 易错点与支持边界

此处配对约定为相邻偶奇维；前后半维配对库需要转换。通常作用在 Q/K，不作用于 V。

## 复杂度

O(NTd) 时间，角度表 O(Td)。

## 接口与依赖

`apply_rope(x: Tensor, offset: int=0, base: float=10000.0) -> Tensor`

x [...,T,d] -> 同形状；d 必须为偶数，位置从 offset 开始。

仓库内依赖：无；可单独复制 implementation.py，仅依赖标准库/PyTorch。

## 运行与验证

在仓库根目录运行：

```bash
python -m attention.rotary.implementation
python -m pytest attention/rotary/test_module.py
```

测试关注：

- `test_norm_relative_positions_and_cached_offsets`

## 面试追问与练习

为什么共同平移 Q/K 的位置不改变 attention score？RoPE 与 KV cache 如何配合？

[独立手写题与起始接口](../../practice/prompts/attention__rotary.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
