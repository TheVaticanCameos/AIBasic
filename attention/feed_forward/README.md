# FFN 与 SwiGLU

优先级：**P0**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

FFN=Wdown*GELU(Wup*x)；SwiGLU=Wdown*(SiLU(Wgate*x)⊙Wvalue*x)。

## Tensor 约定

[B,T,D] → hidden:[B,T,F] → [B,T,D]。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

升维投影、非线性或门控、降维投影；所有位置共享参数但独立计算。

## 易错点与支持边界

SwiGLU 的 up 一次输出 2F 后切分；同 F 下比普通 FFN 多一组投影，不应直接认为参数量相等。

## 复杂度

FFN O(BTDF)，SwiGLU 多一组 D×F 投影；激活 O(BTF)。

## 接口与依赖

`forward(self, x: Tensor) -> Tensor`

x [B,T,D] -> [B,T,D]；hidden [B,T,F]。

仓库内依赖：无；可单独复制 implementation.py，仅依赖标准库/PyTorch。

## 运行与验证

在仓库根目录运行：

```bash
python -m attention.feed_forward.implementation
python -m pytest attention/feed_forward/test_module.py
```

测试关注：

- `test_tokens_independent_and_swiglu_all_projections_trainable`

## 面试追问与练习

FFN 为什么不混合 token？SwiGLU 如何控制参数预算？

[独立手写题与起始接口](../../practice/prompts/attention__feed_forward.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
