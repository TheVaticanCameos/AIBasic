# 跨模块对比

## 归一化

| 方法 | 归约范围 | train/eval 差异 | 本库实现 |
|---|---|---|---|
| BatchNorm | 同通道跨 batch/空间 | 有，使用 running stats | channel-first，支持 2D/3D/4D 输入 |
| LayerNorm | 单样本的最后特征轴 | 无 | normalized_shape=(D,) |
| RMSNorm | 最后特征轴均方，不减均值 | 无 | scale，无 bias |
| GroupNorm | 单样本组内通道/空间 | 无 | 通道可整除 groups |

## Attention

| 变体 | Q heads | KV heads | 关键差异 |
|---|---|---|---|
| MHA | H | H | 每个 query head 有对应 KV |
| GQA | H | 介于 1 与 H | 多个 query heads 共享一组 KV |
| MQA | H | 1 | 全部 query heads 共享 KV |
| Cross-Attention | 可使用上述任意分组 | 同左 | Q 来自 x，KV 来自独立 context |

本库 GQA/MQA 为清晰起见显式 repeat KV，不是 fused kernel 的内存优化实现。KV cache 保存投影后的历史 KV，RoPE/绝对编码的位置信息必须和缓存位置一致。

## 强化学习

| 算法 | 数据 | 动作 | 训练信号 | 核心新增点 |
|---|---|---|---|---|
| REINFORCE | on-policy | 本库离散 | Monte Carlo return | 完整轨迹的 log-prob 加权 |
| 一步 Actor-Critic | on-policy | 本库离散 | TD advantage | value baseline |
| DQN | off-policy replay | 离散 | TD target | target network |
| Double DQN | 同 DQN | 离散 | 解耦选动作与估值 | 缓解 max 过估计 |
| PPO | on-policy rollout | 本库离散 | GAE | ratio clipping、多轮更新 |
| DDPG | off-policy replay | 连续 | TD + deterministic PG | target actor/critic |
| TD3 | 同 DDPG | 连续 | clipped double Q | 双 Q、目标平滑、延迟更新 |
| SAC | off-policy replay | 连续 | entropy-regularized TD | 随机重参数策略、温度 |

Dueling 是 Q 网络结构变化，可以与 Double DQN 同时使用。Actor-Critic 是方法类别，本库的同名单元具体指一步 on-policy 更新。

## 损失与优化器

- CE：互斥类别，target 通常为一个类别索引；BCE：每个标签独立的二分类。
- MSE：大误差梯度随误差增长；Huber：大误差使用线性惩罚。
- KL：有方向；InfoNCE：本库把 batch 内同索引对作为正样本。
- Adam 的 L2 项进入矩估计；AdamW 衰减直接作用于参数。

