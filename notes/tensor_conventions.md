# Tensor 形状与类型约定

## 常见符号

| 符号 | 含义 |
|---|---|
| B | batch size |
| T / Tq / Tk | 序列长度或时间步；query/key 长度 |
| D / d | 模型维度 / 每 head 维度 |
| H / Hkv | attention heads / KV heads；CNN 中 H 表示图像高度 |
| C / I / O | 通道、输入维、输出维；RL 中 O 表示观测维 |
| A | 离散动作数或连续动作维度，以接口为准 |
| N | 展平后的样本数，rollout 中常为 T*B |
| [] | scalar Tensor，有可能参与 autograd |

同一个符号在不同领域可能不同，所以每个接口都有局部说明。

## 类型注释与 shape 注释互补

`Tensor` 本身不编码形状。`actions: Tensor` 仍需说明 `int64 [B]` 或 `floating [B,A]`。尽量不引入额外类型库，以便在机考环境复制使用。

默认 float32；精度对照与 gradcheck 使用 float64。Tensor 之间通常要求 device/dtype 匹配；参数随 nn.Module 的 `.to(...)` 移动。当前 RL algorithm 容器按 CPU 使用，不能像 nn.Module 一样直接 `.to(device)`。

## 关键模式

```text
Q/K/V 投影: [B,T,D] → [B,T,H,d] → [B,H,T,d]
Attention score: [B,H,Tq,d] @ [B,H,d,Tk] → [B,H,Tq,Tk]
合并 heads: [B,H,T,d] → [B,T,H,d] → [B,T,D]
PPO flatten: [T,B,O] → [T*B,O]，所有字段必须以同一顺序展平
DQN gather: [B,A] + index [B,1] → [B,1] → [B]
```

`transpose` 通常只改变 stride；`reshape` 必要时复制；`view` 要求内存布局兼容。`squeeze(-1)` 保留 batch 维，`squeeze()` 会在 B=1 时误删它。

广播从最右侧对齐。`[B,1]-[B]` 得到 `[B,B]`，这常导致 value loss 悄悄算错。

Attention bool mask 统一使用 **True=允许参与**。padding valid mask `[B,Tk]` 先转 `[B,1,1,Tk]`，再与 causal `[Tq,Tk]` 用 `&` 合并。PyTorch MHA 的 `key_padding_mask` 语义相反，做对照时传 `~valid`。

