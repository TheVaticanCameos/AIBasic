# Discounted Return

优先级：**P0**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

G_t=r_t+γ*G_(t+1)；真正终止后续为 0，截断处替换为 V(next_obs)。

## Tensor 约定

rewards/next_values:[T,B]；terminated/truncated:bool [T,B]；输出 [T,B]。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

从后往前扫描；rollout 尾部以 next_value 初始化；截断时 bootstrap 并阻断跨 episode 回传。

## 易错点与支持边界

next_values 必须来自 reset 前下一观测；Monte Carlo 不带价值时传全零。terminated 与 truncated 同时为真时终止优先。

## 复杂度

O(TB) 时间，O(TB) 输出和 O(B) 递推状态。

## 接口与依赖

`discounted_returns(rewards: Tensor, terminated: Tensor, truncated: Tensor, next_values: Tensor, gamma: float=0.99) -> Tensor`

全部输入 [T,B]，flags 为 bool，其余 floating -> returns [T,B]。

next_values[t] 是该 transition 的真实下一观测价值，不能用 reset 后观测。
terminated 不 bootstrap；truncated bootstrap 但阻断跨 episode 递推。
数组末端也用 next_values[-1] bootstrap；纯 Monte Carlo 传全零即可。

仓库内依赖：无；可单独复制 implementation.py，仅依赖标准库/PyTorch。

## 运行与验证

在仓库根目录运行：

```bash
python -m rl.returns.discounted.implementation
python -m pytest rl/returns/discounted/test_module.py
```

测试关注：

- `test_termination_truncation_and_rollout_bootstrap`

## 面试追问与练习

采集步数耗尽与 MDP 真正终止有什么区别？return 与 reward 有何区别？

[独立手写题与起始接口](../../../practice/prompts/rl__returns__discounted.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
