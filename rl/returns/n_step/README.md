# n-step Return

优先级：**P1**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

G_t=Σ_(k=0..m-1) γ^k*r_(t+k)+γ^m*V(next)，m 不超过 n 或可用连续轨迹长度。

## Tensor 约定

输入 [T,B] → target [T,B]；flags 为 bool。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

对每个起点累加至 n 步；遇 episode 边界停止；真正终止去掉 bootstrap。

## 易错点与支持边界

样本必须按时间连续，不能直接把 replay 随机 batch 当成时间序列。截断与终止的 bootstrap 不同。

## 复杂度

教学嵌套循环 O(TnB)，输出 O(TB)。

## 接口与依赖

`n_step_returns(rewards: Tensor, next_values: Tensor, terminated: Tensor, truncated: Tensor, n: int=3, gamma: float=0.99) -> Tensor`

输入均 [T,B]，flags bool -> [T,B]；教学版 O(T*n*B)。

仓库内依赖：无；可单独复制 implementation.py，仅依赖标准库/PyTorch。

## 运行与验证

在仓库根目录运行：

```bash
python -m rl.returns.n_step.implementation
python -m pytest rl/returns/n_step/test_module.py
```

测试关注：

- `test_one_step_and_full_horizon_match_independent_algorithms`
- `test_two_step_hand_calculation`

## 面试追问与练习

n=1 与单步 TD 如何对应？n 增大对偏差与方差有何影响？

[独立手写题与起始接口](../../../practice/prompts/rl__returns__n_step.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
