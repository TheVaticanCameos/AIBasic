# Scaled Dot-Product Attention

优先级：**P0**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

Attention(Q,K,V)=softmax(QKᵀ/sqrt(d)+mask)V。

## Tensor 约定

Q:[...,Tq,d]，K:[...,Tk,d]，V:[...,Tk,dv]；scores:[...,Tq,Tk]。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

计算 score、应用 bool mask、softmax、dropout、加权求和；全遮挡行显式置零。

## 易错点与支持边界

本库 bool mask=True 表示允许。mask 必须可广播。全遮挡行先避免 softmax NaN，再将权重清零。

## 复杂度

每 head O(Tq*Tk*(d+dv))；显式 attention matrix O(Tq*Tk)。

## 接口与依赖

`attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None=None, dropout_p: float=0.0, training: bool=False) -> tuple[Tensor, Tensor]`

q [...,Tq,d], k [...,Tk,d], v [...,Tk,dv] -> output [...,Tq,dv], weights [...,Tq,Tk]。

mask: bool，可广播到 [...,Tq,Tk]，True=可见。全被遮挡行定义为零输出。
返回的 weights 是 dropout 之后的权重，训练时行和不保证等于 1。

`causal_mask(query_length: int, key_length: int, offset: int=0, device: torch.device | None=None) -> Tensor`

返回 bool [Tq,Tk]；query 的绝对位置为 offset + arange(Tq)。

仓库内依赖：无；可单独复制 implementation.py，仅依赖标准库/PyTorch。

## 运行与验证

在仓库根目录运行：

```bash
python -m attention.scaled_dot_product.implementation
python -m pytest attention/scaled_dot_product/test_module.py
```

测试关注：

- `test_sdpa_output_and_all_gradients`
- `test_fully_masked_rows_have_finite_zero_outputs_and_gradients`
- `test_causal_attention_ignores_future`

## 面试追问与练习

为什么除 sqrt(d)？causal mask 与 padding mask 如何合并？全 mask 行怎么定义？

[独立手写题与起始接口](../../practice/prompts/attention__scaled_dot_product.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
