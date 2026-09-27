# 学习路线与掌握进度

标准代码的实现状态与个人掌握程度分开维护。所有“待复习”均是留给你的记录，不代表标准模块未实现。

## 推荐顺序

1. 张量与 autograd → stable ops → Linear → 分类/回归损失 → SGD/Adam → 训练循环。
2. LayerNorm/BatchNorm/Dropout → 卷积/池化/Residual → Attention/MHA → 位置编码/FFN/Transformer。
3. Return/TD → Categorical policy → policy gradient → REINFORCE/Actor-Critic → Replay/target → DQN。
4. GAE/Rollout → PPO objective → PPO；Gaussian/Continuous Q → DDPG → TD3 → SAC。
5. 按薄弱点补齐 P1：RMSNorm/GroupNorm、循环单元、RoPE/GQA/KV Cache、对比损失、n-step、AMP 与 scheduler。

每个模块按“读公式 → 推 shape → 跑示例 → 阅读测试 → 关答案手写 → 追问解释 → 登记错误”完成一轮。
可自行安排首次完成后的 1/3/7/14 天回顾，间隔以实际掌握情况调整。

## 模块进度

| 模块 | 优先级 | 仓库状态 | 能讲原理 | 独立手写 | 练习通过 | 下次复习 |
|---|---|---|---|---|---|---|
| [张量操作与形状变换](fundamentals/tensor_ops/README.md) | P0 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [Autograd 与自定义 backward](fundamentals/autograd/README.md) | P0 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [稳定 softmax / log-softmax / logsumexp](fundamentals/stable_ops/README.md) | P0 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [Linear 全连接层](layers/linear/README.md) | P0 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [Inverted Dropout](layers/dropout/README.md) | P0 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [LayerNorm](layers/layer_norm/README.md) | P0 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [BatchNorm](layers/batch_norm/README.md) | P0 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [RMSNorm](layers/rms_norm/README.md) | P1 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [GroupNorm](layers/group_norm/README.md) | P1 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [Embedding 查表](layers/embedding/README.md) | P1 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [Conv2d 循环与 unfold 实现](layers/conv2d/README.md) | P0 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [Max / Average Pooling](layers/pooling/README.md) | P0 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [Residual Block](layers/residual/README.md) | P0 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [RNN / LSTM / GRU Cell](layers/recurrent/README.md) | P1 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [Scaled Dot-Product Attention](attention/scaled_dot_product/README.md) | P0 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [MHA / GQA / MQA / Cross-Attention](attention/multi_head/README.md) | P0 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [正弦位置编码](attention/positional/README.md) | P0 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [RoPE 旋转位置编码](attention/rotary/README.md) | P1 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [FFN 与 SwiGLU](attention/feed_forward/README.md) | P0 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [Pre-LN Transformer Block](attention/transformer/README.md) | P0 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [推理 KV Cache](attention/kv_cache/README.md) | P1 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [交叉熵、Label Smoothing、BCE with Logits](losses/classification/README.md) | P0 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [MSE 与 Huber](losses/regression/README.md) | P0 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [KL Divergence](losses/divergence/README.md) | P1 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [双向 InfoNCE](losses/contrastive/README.md) | P1 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [SGD 与 Momentum](optimizers/sgd/README.md) | P0 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [Adam 与 AdamW](optimizers/adam/README.md) | P0 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [训练循环、梯度累积与裁剪](training/loop/README.md) | P0 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [Warmup + Cosine 学习率计划](training/schedules/README.md) | P1 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [AMP 混合精度训练](training/mixed_precision/README.md) | P1 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [Discounted Return](rl/returns/discounted/README.md) | P0 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [TD Target 与优势标准化](rl/returns/td/README.md) | P0 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [Generalized Advantage Estimation](rl/returns/gae/README.md) | P0 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [n-step Return](rl/returns/n_step/README.md) | P1 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [Replay Buffer](rl/buffers/replay/README.md) | P0 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [Rollout Buffer](rl/buffers/rollout/README.md) | P0 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [Categorical Policy 与 Value Network](rl/policies/categorical/README.md) | P0 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [Gaussian Policy 与 Tanh 修正](rl/policies/gaussian/README.md) | P1 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [确定性策略与 Continuous Q](rl/policies/continuous/README.md) | P1 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [Policy Gradient Loss](rl/objectives/policy_gradient/README.md) | P0 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [PPO Clipped Objective](rl/objectives/ppo/README.md) | P0 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [目标网络同步与参数冻结](rl/updates/target/README.md) | P0 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [REINFORCE 更新](rl/algorithms/reinforce/README.md) | P0 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [一步 Actor-Critic](rl/algorithms/actor_critic/README.md) | P0 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [DQN / Double DQN / Dueling](rl/algorithms/dqn/README.md) | P0 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [PPO 算法组装](rl/algorithms/ppo/README.md) | P0 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [DDPG](rl/algorithms/ddpg/README.md) | P1 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [TD3](rl/algorithms/td3/README.md) | P1 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |
| [Soft Actor-Critic](rl/algorithms/sac/README.md) | P1 | 已实现、有测试 | 待复习 | 待复习 | 待复习 | — |

## 扩展方向（未纳入当前实现）

P2 留待岗位需要时加入：RLHF/偏好优化、扩散模型、GNN、分布式训练、FlashAttention kernel。当前代码覆盖本方案 P0/P1 的教学核心，支持边界见各模块 README。
