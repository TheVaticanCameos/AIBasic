# TD3

优先级：**P1**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

target=r+γ min(Q1target,Q2target)(next,clip(actor_target(next)+noise))。

## Tensor 约定

obs:[B,O]，actions/noise:[B,A]，双 Q:[B]。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

每步更新双 critic；目标动作加截断噪声；每 policy_delay 步更新 actor 与全部 targets。

## 易错点与支持边界

目标平滑噪声与行为探索噪声用途不同。actor 仅使用 Q1；延迟步日志 actor_loss=None 表示本轮未更新。

## 复杂度

双 critic 开销；actor 与 target 更新频率降低。

## 接口与依赖

`act(self, observations: Tensor, noise_std: float=0.0) -> Tensor`

obs [B,O] -> actions [B,A]，范围 [-1,1]。

`targets(self, batch: TransitionBatch) -> Tensor`

返回 [B]；目标噪声与探索噪声相互独立。

`update(self, batch: TransitionBatch) -> dict[str, float | None]`

每次更新 critic；每 policy_delay 次更新 actor 与所有 targets。

仓库内依赖：[rl.buffers.replay.implementation](../../buffers/replay/implementation.py)、[rl.policies.continuous.implementation](../../policies/continuous/implementation.py)、[rl.returns.td.implementation](../../returns/td/implementation.py)、[rl.updates.target.implementation](../../updates/target/implementation.py)

## 运行与验证

在仓库根目录运行：

```bash
python -m rl.algorithms.td3.implementation
python -m pytest rl/algorithms/td3/test_module.py
```

测试关注：

- `test_delayed_actor_and_target_updates`
- `test_clipped_double_q_target_and_terminal_mask`

## 面试追问与练习

TD3 的三个改进分别针对什么问题？为何取两个 critic 的最小值？

[独立手写题与起始接口](../../../practice/prompts/rl__algorithms__td3.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
