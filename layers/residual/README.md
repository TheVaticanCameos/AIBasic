# Residual Block

优先级：**P0**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

y=ReLU(F(x)+shortcut(x))，F=Conv-BN-ReLU-Conv-BN。

## Tensor 约定

输入 [B,C,H,W]；输出 [B,O,ceil(H/S),ceil(W/S)]。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

主分支两次卷积；通道或 stride 变化时 shortcut 用 1×1 投影与 BN 对齐。

## 易错点与支持边界

残差相加要求完整 shape 一致；此处采用 basic block 和 post-add ReLU，未实现 bottleneck。

## 复杂度

主开销为两个 3×3 卷积；投影分支增加 1×1 卷积参数。

## 接口与依赖

`forward(self, x: Tensor) -> Tensor`

x [B,C,H,W] -> [B,O,ceil(H/S),ceil(W/S)]。

仓库内依赖：[layers.batch_norm.implementation](../batch_norm/implementation.py)、[layers.conv2d.implementation](../conv2d/implementation.py)

## 运行与验证

在仓库根目录运行：

```bash
python -m layers.residual.implementation
python -m pytest layers/residual/test_module.py
```

测试关注：

- `test_identity_path_and_projection`

## 面试追问与练习

残差为何利于梯度传播？将主分支置零后输出是否总等于原输入？

[独立手写题与起始接口](../../practice/prompts/layers__residual.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
