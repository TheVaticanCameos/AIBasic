# 综合组装题

这些题目可以复用已经独立手写过的模块。目标是解释数据流与更新顺序，而非背下一份完整工程。

## 1. 微型语言模型（45–60 分钟）

输入 token `[B,T]`，输出 logits `[B,T,V]`。组合 Embedding、位置编码、causal Transformer、Linear 和 CE。准备模 V 递增序列，目标右移一位。

验收：禁止未来信息泄漏；loss 使用展平的 `[B*T,V]` 与 `[B*T]`；小训练集 loss 下降。参考 [监督学习示例](../examples/supervised.py)。

## 2. Double DQN 一次更新（30–45 分钟）

输入 TransitionBatch，完成 online 选动作、target 估值、gather 当前 Q、Huber、优化器更新、target 软更新。

验收：用两个故意排序不同的 Q 网络区分 DQN/Double DQN；target 无梯度；时间限制截断仍 bootstrap。参考 [DQN](../rl/algorithms/dqn/implementation.py)。

## 3. PPO rollout 到多轮更新（60–90 分钟）

给定 `[T,B,...]` rollout，完成 GAE、统一 flatten、优势标准化、minibatch shuffle、policy/value/entropy loss。

验收：两个 epoch 后 old_log_prob 不变；B=1 正确；跨 episode 的 return 不串联；非整除 minibatch 可运行。参考 [PPO](../rl/algorithms/ppo/implementation.py)。

## 4. SAC actor 与温度（45–60 分钟）

给定双 critic 与高斯 actor，完成重参数采样、tanh Jacobian、actor loss 和 log_alpha 更新。

验收：actor 获得动作路径梯度；critic 参数不因 actor loss 新增梯度；alpha 始终为正；动作接近 ±1 时 log-prob 有限。参考 [SAC](../rl/algorithms/sac/implementation.py)。
