# 正弦位置编码

优先级：**P0**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

PE(pos,2i)=sin(pos/10000^(2i/D))，PE(pos,2i+1)=cos(...)。

## Tensor 约定

输出 [T,D]，与 token embeddings [B,T,D] 广播相加。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

构造位置与逆频率外积，交替写 sin/cos；支持 offset 和奇数 D。

## 易错点与支持边界

生成阶段 offset 必须跟累计位置一致。奇数 D 的最后一个 sin 不存在对应 cos。

## 复杂度

O(TD) 时间与输出内存，无可训练参数。

## 接口与依赖

`sinusoidal_encoding(length: int, dim: int, offset: int=0, device: torch.device | None=None, dtype: torch.dtype=torch.float32) -> Tensor`

返回 [T,D]，与 token embedding [B,T,D] 广播相加。

仓库内依赖：无；可单独复制 implementation.py，仅依赖标准库/PyTorch。

## 运行与验证

在仓库根目录运行：

```bash
python -m attention.positional.implementation
python -m pytest attention/positional/test_module.py
```

测试关注：

- `test_zero_position_and_offset_for_odd_dimensions`

## 面试追问与练习

为什么 Attention 需要位置信息？绝对与相对位置编码的差别是什么？

[独立手写题与起始接口](../../practice/prompts/attention__positional.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
