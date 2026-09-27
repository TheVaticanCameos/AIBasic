# TD Target 与优势标准化

优先级：**P0**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

target=r+γ*(1-terminated)*V(next)；A_norm=(A-mean(A))/(std(A)+eps)。

## Tensor 约定

rewards/next_values/terminated:[B]；输出 target:[B]；标准化支持任意同形状。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

在 no_grad 中计算 TD 监督目标；优势使用 correction=0 的标准差。

## 易错点与支持边界

时间限制截断仍 bootstrap；不能将 terminated|truncated 作为 TD mask。单样本优势标准化为零。

## 复杂度

O(B) 时间与输出。

## 接口与依赖

`td_target(rewards: Tensor, next_values: Tensor, terminated: Tensor, gamma: float=0.99) -> Tensor`

rewards/next_values [B]，terminated bool [B] -> detached target [B]。

时间限制 truncated 不传入 terminated，仍需要 bootstrap。

`normalize_advantages(advantages: Tensor, eps: float=1e-08) -> Tensor`

advantages [...] -> 同形状；总体标准差，单样本也不会 NaN。

仓库内依赖：无；可单独复制 implementation.py，仅依赖标准库/PyTorch。

## 运行与验证

在仓库根目录运行：

```bash
python -m rl.returns.td.implementation
python -m pytest rl/returns/td/test_module.py
```

测试关注：

- `test_bootstrap_and_gradient_boundary`

## 面试追问与练习

为什么 target 要 detach？TD 与 Monte Carlo 的偏差/方差有什么区别？

[独立手写题与起始接口](../../../practice/prompts/rl__returns__td.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
