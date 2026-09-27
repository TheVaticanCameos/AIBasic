# Soft Actor-Critic

优先级：**P1**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

target=r+γ(min Qtarget(next,a_next)-α logπ(a_next))；actor loss=mean(αlogπ-min Q)。

## Tensor 约定

obs:[B,O]、actions:[B,A]、Q/log_prob:[B]；log_alpha/alpha:[]。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

重参数采样与 tanh 修正；更新双 critic、actor、可选自动温度；软更新 Q targets。

## 易错点与支持边界

现代版本无单独 value 网络；actor 更新时 critic 参数冻结；alpha 不接收 actor/critic loss 梯度。目标熵默认 -A。

## 复杂度

双 critic、actor 和目标 Q 前向/反向；温度更新是标量。

## 接口与依赖

`alpha(self) -> Tensor`

温度 scalar []；策略与 critic 优化不应更新 alpha。

`act(self, observations: Tensor, deterministic: bool=False) -> Tensor`

obs [B,O] -> 标准动作 [B,A]，范围 [-1,1]。

`targets(self, batch: TransitionBatch) -> Tensor`

返回 entropy-regularized TD target [B]。

`update(self, batch: TransitionBatch) -> dict[str, float]`

obs [B,O], actions [B,A]；依次更新 critic、actor、温度和 target。

仓库内依赖：[rl.buffers.replay.implementation](../../buffers/replay/implementation.py)、[rl.policies.continuous.implementation](../../policies/continuous/implementation.py)、[rl.policies.gaussian.implementation](../../policies/gaussian/implementation.py)、[rl.returns.td.implementation](../../returns/td/implementation.py)、[rl.updates.target.implementation](../../updates/target/implementation.py)

## 运行与验证

在仓库根目录运行：

```bash
python -m rl.algorithms.sac.implementation
python -m pytest rl/algorithms/sac/test_module.py
```

测试关注：

- `test_sac_updates_actor_and_temperature_without_target_gradients`
- `test_fixed_temperature_and_terminal_targets`
- `test_soft_target_uses_minimum_q_and_subtracts_log_prob`

## 面试追问与练习

熵正则如何影响探索？为什么优化 log_alpha？温度 loss 的策略项为何 detach？

[独立手写题与起始接口](../../../practice/prompts/rl__algorithms__sac.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
