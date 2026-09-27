# 推理 KV Cache

优先级：**P1**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

Kcache=concat(Kpast,Knew)，Vcache 同理；新 Q 关注所有合法过去 key。

## Tensor 约定

新输入 [B,Tnew,D]；缓存 [B,Hkv,Tpast,d]；mask [Tnew,Tpast+Tnew]。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

复用 MHA 权重投影新块、追加 KV、带 offset 构造 causal mask、只输出新 token。

## 易错点与支持边界

只允许 eval + no_grad。换序列须 clear；batch 顺序不能变化。不集成 RoPE 或 padding mask，输入应已带绝对位置编码。

## 复杂度

缓存 O(B*Hkv*T*d)，单 token 注意力 O(B*H*T*d)，追加 cat 每次复制已有缓存（教学简化）。

## 接口与依赖

`cached_attention(module: MultiHeadAttention, x: Tensor, cache: KVCache) -> Tensor`

x [B,Tnew,D] -> [B,Tnew,D]；按时间顺序调用，换序列必须 clear。

输入应已包含绝对位置编码。本函数不集成 RoPE 或 padding mask。

`clear(self) -> None`

见实现中的类型与形状注释。

仓库内依赖：[attention.multi_head.implementation](../multi_head/implementation.py)、[attention.scaled_dot_product.implementation](../scaled_dot_product/implementation.py)、[fundamentals.tensor_ops.implementation](../../fundamentals/tensor_ops/implementation.py)

## 运行与验证

在仓库根目录运行：

```bash
python -m attention.kv_cache.implementation
python -m pytest attention/kv_cache/test_module.py
```

测试关注：

- `test_chunked_decoding_matches_full_sequence`

## 面试追问与练习

为什么只缓存 KV？生成多个 token 的 chunk 为什么不能简单使用左上角三角 mask？

[独立手写题与起始接口](../../practice/prompts/attention__kv_cache.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
