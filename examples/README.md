# 端到端示例

这些入口展示如何把单元组合起来，核心算法保留在相应学习目录。

| 入口 | 数据/环境 | 流程 |
|---|---|---|
| `supervised --model mlp` | 合成二维判别规则嵌入四维输入 | Linear → ReLU → Linear → CE → 手写 Adam |
| `supervised --model cnn` | 8×8 水平/垂直线图案 | ResidualBlock → flatten → Linear → CE |
| `supervised --model transformer` | 模 8 递增 token 序列 | Embedding + PE → causal block → Linear → next-token CE |
| `cartpole --algorithm dqn` | CartPole-v1 | epsilon-greedy → replay → Double DQN → soft update |
| `cartpole --algorithm ppo` | CartPole-v1 | rollout 128 步 → GAE → 多轮 minibatch PPO |
| `cartpole --algorithm reinforce` | CartPole-v1 | 完整 episode → MC return → REINFORCE |
| `cartpole --algorithm actor_critic` | CartPole-v1 | 单步采样 → detached TD target → actor/value 更新 |
| `pendulum --algorithm ddpg/td3/sac` | Pendulum-v1 | 标准动作 → 环境缩放 → replay → 连续控制更新 |

监督学习例子固定小训练集，验证 loss 下降与拟合能力，未提供 train/validation 数据划分。语言模型例子学习人工 next-token 规则，不是自然语言建模能力评测。

CartPole 的评估使用独立环境与固定评估种子，选择 argmax 动作；训练中仍按算法探索。PPO 在预算末尾也消费剩余 rollout；REINFORCE 只在 episode 结束时更新，预算末尾残缺轨迹舍弃。

Pendulum 前 128 步随机探索。Replay 存储标准 [-1,1] 动作，环境交互使用 `low + (a+1)/2*(high-low)`。SAC 熵与 log-prob 定义在标准动作空间，目标熵同样采用这个约定。

环境返回的真实 `next_obs` 先用于 buffer/value，再执行 reset。所有环境均通过 finally 关闭；不启动渲染窗口、不下载数据或权重、不保存训练产物。

运行命令见根目录 README；`--help` 展示参数。想观察 RL 学习曲线，可增加预算并自行记录多种子数据；短测试只检查交互和更新流程。

