# DQN / Double DQN / Dueling

优先级：**P0**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

DQN target=r+γ max_a Qtarget(next,a)；Double 用 online argmax 选动作、target 估值。

## Tensor 约定

obs:[B,O] → Q:[B,A]；actions:[B] gather 后为 [B]。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

epsilon-greedy 采样、replay batch、计算 detached target、Huber 更新、soft target update；Dueling 可独立启用。

## 易错点与支持边界

Dueling: Q=V+A-mean_action(A)，与 Double DQN 是独立改进。这里目标网络采用逐步软更新，不是固定间隔硬复制。

## 复杂度

每更新为数次 Q 网络前向和一次反向；replay 独立占用内存。

## 接口与依赖

`forward(self, observations: Tensor) -> Tensor`

obs [B,O] -> Q [B,A]；dueling 的 advantage 按动作维减均值。

`act(self, observations: Tensor, epsilon: float=0.0) -> Tensor`

obs [B,O] -> int64 actions [B]；每个样本独立 epsilon-greedy。

`targets(self, batch: TransitionBatch) -> Tensor`

返回 detached [B]；Double DQN 用 online 选动作、target 估值。

`update(self, batch: TransitionBatch) -> float`

obs [B,O], action [B]；一次优化后软更新 target。

仓库内依赖：[fundamentals.tensor_ops.implementation](../../../fundamentals/tensor_ops/implementation.py)、[losses.regression.implementation](../../../losses/regression/implementation.py)、[rl.buffers.replay.implementation](../../buffers/replay/implementation.py)、[rl.returns.td.implementation](../../returns/td/implementation.py)、[rl.updates.target.implementation](../../updates/target/implementation.py)

## 运行与验证

在仓库根目录运行：

```bash
python -m rl.algorithms.dqn.implementation
python -m pytest rl/algorithms/dqn/test_module.py
```

测试关注：

- `test_double_dqn_selection_terminal_and_target_gradients`
- `test_dueling_value_is_mean_q_and_greedy_actions`

## 面试追问与练习

Double DQN 如何缓解过估计？Dueling 为什么要对 advantage 减均值？

[独立手写题与起始接口](../../../practice/prompts/rl__algorithms__dqn.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
