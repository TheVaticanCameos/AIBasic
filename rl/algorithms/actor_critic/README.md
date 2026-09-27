# 一步 Actor-Critic

优先级：**P0**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

target=r+γV(next)；A=target-V(s)；actor 用 -logπ*A，critic 拟合 target。

## Tensor 约定

TransitionBatch：obs:[B,O]、离散 actions:[B]、rewards/flags:[B]。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

用旧 value 算 detached target 与 advantage；策略和价值分别计算 loss、分别优化。

## 易错点与支持边界

必须使用刚采集的 on-policy 数据；此函数不是 replay-based actor-critic，也不是并行 A2C 框架。

## 复杂度

策略与价值各一次前后向。

## 接口与依赖

`actor_critic_update(policy: CategoricalPolicy, value: ValueNetwork, policy_optimizer: torch.optim.Optimizer, value_optimizer: torch.optim.Optimizer, batch: TransitionBatch, gamma: float=0.99, entropy_coef: float=0.01) -> dict[str, float]`

batch 的 obs [B,O]，actions [B]；要求刚采集的 on-policy transition。

仓库内依赖：[losses.regression.implementation](../../../losses/regression/implementation.py)、[rl.buffers.replay.implementation](../../buffers/replay/implementation.py)、[rl.objectives.policy_gradient.implementation](../../objectives/policy_gradient/implementation.py)、[rl.policies.categorical.implementation](../../policies/categorical/implementation.py)、[rl.returns.td.implementation](../../returns/td/implementation.py)

## 运行与验证

在仓库根目录运行：

```bash
python -m rl.algorithms.actor_critic.implementation
python -m pytest rl/algorithms/actor_critic/test_module.py
```

测试关注：

- `test_terminal_reward_learning_and_no_next_obs_gradient`

## 面试追问与练习

与 REINFORCE 相比偏差/方差怎样变化？为何不能把 value 梯度混入 actor loss？

[独立手写题与起始接口](../../../practice/prompts/rl__algorithms__actor_critic.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
