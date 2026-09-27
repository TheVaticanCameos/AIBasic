# RMSNorm

优先级：**P1**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

y=x/sqrt(mean(x²)+eps)*γ。

## Tensor 约定

x:[...,D]；均方与缩放:[...,1]；γ:[D]。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

计算最后一维均方；半精度输入升为 float32 累加；缩放后转换回输入 dtype 再 affine。

## 易错点与支持边界

不减均值，不使用 bias；不能把它当作输出均值为零的 LayerNorm。

## 复杂度

O(ND) 时间，参数 O(D)。

## 接口与依赖

`forward(self, x: Tensor) -> Tensor`

x [...,D] -> [...,D]。

仓库内依赖：无；可单独复制 implementation.py，仅依赖标准库/PyTorch。

## 运行与验证

在仓库根目录运行：

```bash
python -m layers.rms_norm.implementation
python -m pytest layers/rms_norm/test_module.py
```

测试关注：

- `test_rms_formula_and_gradient`

## 面试追问与练习

RMSNorm 省略了什么？平移不变性和缩放行为与 LayerNorm 有何差异？

[独立手写题与起始接口](../../practice/prompts/layers__rms_norm.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
