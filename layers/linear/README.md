# Linear 全连接层

优先级：**P0**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

Y=XWᵀ+b。

## Tensor 约定

X:[...,I]，W:[O,I]，b:[O] → Y:[...,O]。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

注册 weight/bias 为 Parameter；矩阵乘法作用于最后一维；bias 广播。

## 易错点与支持边界

不要将权重写成普通 Tensor，否则 parameters()/optimizer 不会自动发现它。最后一维之外都是批维。

## 复杂度

N 个输入向量时 O(NIO)，参数 O(IO+O)。

## 接口与依赖

`forward(self, x: Tensor) -> Tensor`

x [...,I] -> [...,O]，权重仅作用在最后一维。

仓库内依赖：无；可单独复制 implementation.py，仅依赖标准库/PyTorch。

## 运行与验证

在仓库根目录运行：

```bash
python -m layers.linear.implementation
python -m pytest layers/linear/test_module.py
```

测试关注：

- `test_output_input_and_parameter_gradients`

## 面试追问与练习

为什么 PyTorch 的 weight 是 [out,in]？无 bias 情况如何实现？

[独立手写题与起始接口](../../practice/prompts/layers__linear.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
