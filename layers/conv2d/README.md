# Conv2d 循环与 unfold 实现

优先级：**P0**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

Y[b,o,i,j]=Σ(c,u,v) Xpad[b,c,i*S+u,j*S+v]*W[o,c,u,v]+b[o]。

## Tensor 约定

X:[B,C,H,W]，W:[O,C,K,K] → [B,O,Hout,Wout]；Hout=floor((H+2P-K)/S)+1。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

循环版逐窗口相乘求和；向量版提取 [B,C*K*K,L] 窗口后矩阵乘法。

## 易错点与支持边界

实际计算为互相关，不翻转核。限定方形核、整数 stride/padding、groups=1、dilation=1、合法输出尺寸。

## 复杂度

计算 O(B*O*C*K²*Hout*Wout)；unfold 额外内存 O(B*C*K²*Hout*Wout)。

## 接口与依赖

`conv2d_loop(x: Tensor, weight: Tensor, bias: Tensor | None=None, stride: int=1, padding: int=0) -> Tensor`

x [B,C,H,W], weight [O,C,K,K], bias [O] -> [B,O,Hout,Wout]。

`conv2d_unfold(x: Tensor, weight: Tensor, bias: Tensor | None=None, stride: int=1, padding: int=0) -> Tensor`

同循环版接口；unfold 仅负责提取窗口，卷积计算使用矩阵乘法。

`forward(self, x: Tensor) -> Tensor`

x [B,C,H,W] -> [B,O,Hout,Wout]。

仓库内依赖：无；可单独复制 implementation.py，仅依赖标准库/PyTorch。

## 运行与验证

在仓库根目录运行：

```bash
python -m layers.conv2d.implementation
python -m pytest layers/conv2d/test_module.py
```

测试关注：

- `test_loop_unfold_torch_outputs_and_all_gradients`

## 面试追问与练习

stride/padding 如何影响输出尺寸？unfold 为什么可能更耗内存？

[独立手写题与起始接口](../../practice/prompts/layers__conv2d.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
