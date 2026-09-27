# Pre-LN Transformer Block

优先级：**P0**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

h=x+Dropout(Attn(LN(x)))；y=h+Dropout(FFN(LN(h)))。

## Tensor 约定

输入输出均 [B,T,D]；mask 为 [T,T] 或可广播 [B,1,T,T]。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

先 norm 后子层，再 dropout 与残差；attention/FFN 独立模块组合。

## 易错点与支持边界

causal 属性由 mask 决定，不传 mask 就是双向注意力。没有 cross-attention 子层；完整语言模型在 examples 中组装。

## 复杂度

O(BT D² + BT²D + BTDF)，显式 score 内存 O(BHT²)。

## 接口与依赖

`forward(self, x: Tensor, mask: Tensor | None=None) -> Tensor`

x [B,T,D] -> [B,T,D]；mask bool [T,T] 或 [B,1,T,T]，True=可见。

仓库内依赖：[attention.feed_forward.implementation](../feed_forward/implementation.py)、[attention.multi_head.implementation](../multi_head/implementation.py)、[layers.layer_norm.implementation](../../layers/layer_norm/implementation.py)

## 运行与验证

在仓库根目录运行：

```bash
python -m attention.transformer.implementation
python -m pytest attention/transformer/test_module.py
```

测试关注：

- `test_residual_identity_and_causality`

## 面试追问与练习

Pre-LN 与 Post-LN 的残差路径差在哪里？为什么训练语言模型需要 causal mask？

[独立手写题与起始接口](../../practice/prompts/attention__transformer.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
