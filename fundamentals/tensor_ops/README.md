# 张量操作与形状变换

优先级：**P0**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

reshape/transpose 改变维度组织；gather 按样本挑选动作；einsum 显式表达求和维度。

## Tensor 约定

[B,T,D] ↔ [B,H,T,d]，D=H*d；Q:[B,A] + actions:[B] → [B]。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

先明确维度含义，再拆分 head，交换序列/head 轴；合并时逆序操作。

## 易错点与支持边界

transpose 后 tensor 可能不连续；view 不一定合法，reshape 必要时复制。gather 的索引为 int64。

## 复杂度

head 变换通常 O(BTD)，矩阵乘法 O(BNMD)；reshape 可能分配新内存。

## 接口与依赖

`image_to_tokens(x: Tensor) -> Tensor`

通过 permute 将 `[B,C,H,W]` 转为 `[B,H,W,C]`，再 reshape 为按行优先的 `[B,H*W,C]`。

`split_heads(x: Tensor, num_heads: int) -> Tensor`

x [B,T,D] -> [B,H,T,D/H]。

`merge_heads(x: Tensor) -> Tensor`

x [B,H,T,d] -> [B,T,H*d]；transpose 后使用 reshape。

`select_action_values(q: Tensor, actions: Tensor) -> Tensor`

q floating [B,A]，actions int64 [B] -> 所选动作值 [B]。

`pairwise_dot(x: Tensor, y: Tensor) -> Tensor`

x [B,N,D], y [B,M,D] -> [B,N,M]；演示 einsum。

仓库内依赖：无；可单独复制 implementation.py，仅依赖标准库/PyTorch。

## 运行与验证

在仓库根目录运行：

```bash
python -m fundamentals.tensor_ops.implementation
python -m pytest fundamentals/tensor_ops/test_module.py
```

测试关注：

- `test_heads_roundtrip_noncontiguous_and_gather`
- `test_permute_preserves_spatial_order_and_channels`

## 面试追问与练习

为什么 transpose 后不能总用 view？gather 与高级索引的梯度有什么特点？

[独立手写题与起始接口](../../practice/prompts/fundamentals__tensor_ops.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
