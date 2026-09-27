# MHA / GQA / MQA / Cross-Attention

优先级：**P0**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

各 head 独立 attention，拼接后乘输出投影；Hkv=H 为 MHA，Hkv=1 为 MQA。

## Tensor 约定

x:[B,Tq,D]，context:[B,Tk,D]；Q:[B,H,Tq,d]；K/V:[B,Hkv,Tk,d]。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

分别投影 Q/K/V、拆 head、按组映射 KV、调用 attention、合 head 与输出投影。

## 易错点与支持边界

要求 D%H=0 且 H%Hkv=0。padding mask 从 [B,Tk] 转为 [B,1,1,Tk]。此教学版 repeat_interleave 显式复制 KV。

## 复杂度

注意力 O(B*H*Tq*Tk*d)，score 内存 O(B*H*Tq*Tk)；GQA 投影/KV cache 更小，但本实现展开有额外临时内存。

## 接口与依赖

`forward(self, x: Tensor, context: Tensor | None=None, mask: Tensor | None=None) -> Tensor`

x [B,Tq,D], context [B,Tk,D] -> [B,Tq,D]。

mask bool [Tq,Tk] 或 [B,1,Tq,Tk]；padding mask 应转换为 [B,1,1,Tk]。

仓库内依赖：[attention.scaled_dot_product.implementation](../scaled_dot_product/implementation.py)、[fundamentals.tensor_ops.implementation](../../fundamentals/tensor_ops/implementation.py)

## 运行与验证

在仓库根目录运行：

```bash
python -m attention.multi_head.implementation
python -m pytest attention/multi_head/test_module.py
```

测试关注：

- `test_mha_matches_torch_cross_attention`
- `test_gqa_matches_explicit_per_head_selection`

## 面试追问与练习

Cross-Attention 的 Q 与 KV 分别来自哪里？GQA 为什么节省缓存？

[独立手写题与起始接口](../../practice/prompts/attention__multi_head.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
