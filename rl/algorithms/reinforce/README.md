# REINFORCE 更新

优先级：**P0**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

loss=-mean(logπ(a_t|s_t)*G_t)。

## Tensor 约定

observations:[N,O]，actions:int64 [N]，returns:[N]；N 为轨迹总步数。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

外层采完整轨迹并计算 return；本函数重算当前 log-prob、计算目标、反向与 step。

## 易错点与支持边界

可选标准化仅是启发式，小 batch/恒定 return 会抹去信号。示例优化有限 episode MC 回报；预算结束的残缺轨迹不更新。

## 复杂度

策略网络前后向 O(N*network_cost)。

## 接口与依赖

`reinforce_update(policy: CategoricalPolicy, optimizer: torch.optim.Optimizer, observations: Tensor, actions: Tensor, returns: Tensor, normalize: bool=True) -> float`

obs [N,O], actions int64 [N], returns [N] -> loss；N 为轨迹总步数。

标准化是可选 variance-reduction 启发式；小 batch 下可能抹去学习信号。

仓库内依赖：[rl.objectives.policy_gradient.implementation](../../objectives/policy_gradient/implementation.py)、[rl.policies.categorical.implementation](../../policies/categorical/implementation.py)、[rl.returns.td.implementation](../../returns/td/implementation.py)

## 运行与验证

在仓库根目录运行：

```bash
python -m rl.algorithms.reinforce.implementation
python -m pytest rl/algorithms/reinforce/test_module.py
```

测试关注：

- `test_rewarded_action_probability_increases`

## 面试追问与练习

REINFORCE 为什么方差大？按轨迹求和与按 transition 均值的标度差在哪里？

[独立手写题与起始接口](../../../practice/prompts/rl__algorithms__reinforce.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
