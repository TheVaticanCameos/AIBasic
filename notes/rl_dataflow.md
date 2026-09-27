# RL 数据、边界与梯度流

## 三种边界

| 情况 | bootstrap 下一状态价值 | 跨边界递推 return/GAE |
|---|---|---|
| MDP 真正终止 `terminated=True` | 否 | 否 |
| 外部时间限制 `truncated=True` | 是，使用真实 next_obs | 否，下一条可能已 reset |
| rollout 采集预算结束，episode 未结束 | 是 | 数组外没有后续 residual，trace 在此停止 |

先计算/保存 transition 的真实下一观测，再 reset。使用自动 reset/vector env 时，必须按该环境 API 取 final observation；本仓库示例使用单环境手动 reset。

GAE 中两个 mask 的职责不同：

```text
delta_t = reward_t + gamma * (1-terminated_t) * next_value_t - value_t
adv_t   = delta_t + gamma * lambda * (1-(terminated_t|truncated_t)) * adv_(t+1)
```

单元测试包含“截断后下一条是新 episode 的大 reward”，确保这个 reward 不污染上一条轨迹。

## 梯度边界

| 计算 | 接收梯度 | 不接收梯度 |
|---|---|---|
| DQN TD loss | online Q(s,a) | target network、TD target |
| Policy gradient | 当前 log-prob | return / advantage |
| PPO ratio | new log-prob | old log-prob、GAE、return |
| Actor-Critic value loss | 当前 V(s) | 下一状态价值与 target |
| DDPG/TD3 actor loss | actor → action → critic 的输入路径 | critic 参数 |
| SAC actor loss | 重参数动作、当前 log-prob、actor | critic 参数、alpha |
| SAC temperature loss | log_alpha | 当前策略 log-prob |

`frozen_parameters(critic)` 与 `torch.no_grad()` 不同：前者允许 Q 对 action 的梯度继续传回 actor；后者会把这条路径也切掉。

## PPO 更新顺序

```text
采样时固定 old_log_prob / value / next_value
                 ↓
finish rollout：[T,B,...]
                 ↓
一次性计算 detached GAE / return，优势按整批标准化
                 ↓
统一 flatten 为 [T*B,...]
                 ↓
多轮 shuffle → minibatch → current policy/value → loss → update
                 ↓
丢弃该 rollout，用新策略重新采样
```

PPO 的 clipped loss 没有强制 KL 约束；本实现返回 approximate KL 和 clip fraction，未增加 KL early stop。不要因为 loss 不报错就认为策略更新足够小。

