# Autograd 与自定义 backward

优先级：**P0**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

y=x²；反向传播 grad_x=grad_output*2x。

## Tensor 约定

x、输出、grad_output 均为任意同形状 [...]。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

forward 保存原输入；backward 用上游梯度乘局部导数；测试检查一阶与二阶导数。

## 易错点与支持边界

backward 默认累加到 .grad；detach 切断图但共享存储。no_grad 阻止新操作建图，eval 只切换模块行为。

## 复杂度

逐元素平方 O(N) 时间、保存 O(N) 输入。

## 接口与依赖

`square(x: Tensor) -> Tensor`

x floating [...] -> [...]，使用自定义 autograd Function。

`forward(ctx: FunctionCtx, x: Tensor) -> Tensor`

x [...] -> y [...]；保存 backward 需要的输入。

`backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor`

grad_output [...] -> grad_x [...]；链式法则。

仓库内依赖：无；可单独复制 implementation.py，仅依赖标准库/PyTorch。

## 运行与验证

在仓库根目录运行：

```bash
python -m fundamentals.autograd.implementation
python -m pytest fundamentals/autograd/test_module.py
```

测试关注：

- `test_backward_and_double_backward`
- `test_accumulation_detach_and_no_grad`

## 面试追问与练习

detach 与 no_grad 有何不同？非叶子张量为什么默认看不到 .grad？

[独立手写题与起始接口](../../practice/prompts/fundamentals__autograd.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
