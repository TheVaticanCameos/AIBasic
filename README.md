# AIBasic：PyTorch 深度学习与强化学习复习库

面向通用算法岗的机考与面试准备。以 **49 个独立学习单元**组织 P0/P1 内容：每个单元包含手写实现、中文讲解、最小运行示例和正确性测试。相关变体（如 RNN/LSTM/GRU、Adam/AdamW）放在同一单元，分别提供可独立导入的类。

目标是能解释公式、推导 Tensor 形状、独立手写，并检查数值与梯度。默认 CPU、小张量；深度学习示例使用合成数据，无需下载数据集。强化学习环境通过可选 Gymnasium 依赖提供。

## 开始使用

建议 Python 3.10+、PyTorch 2.4+。这是声明的 API 下限；本次实际验证环境与结果见 [VALIDATION.md](VALIDATION.md)。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,rl]'

# 一个学习单元：先读 README，再运行实现与测试
python -m attention.multi_head.implementation
python -m pytest attention/multi_head/test_module.py

# 全量验证
python -m pytest
ruff check .
mypy fundamentals layers attention losses optimizers training rl examples
```

如果已有合适的 PyTorch 环境，可以直接在该环境安装；本地交付使用 `.venv` 并复用机器已有的 CPU PyTorch。仅运行模块和监督学习示例时不需要 `rl` extra；未安装 Gymnasium 时，RL 环境集成测试会明确 skip，纯张量 RL 测试仍可运行。

所有命令从仓库根目录执行。推荐 `python -m 路径.implementation`，这样组合模块的导入路径一致。无仓库内依赖的单元也可以直接复制 `implementation.py` 单独使用；组合单元的 README 会列出依赖。

## 内容导航

完整索引与个人复习进度见 [ROADMAP.md](ROADMAP.md)。

| 目录 | 学习内容 | 建议入口 |
|---|---|---|
| `fundamentals/` | 张量操作、稳定计算、autograd | [张量与形状](fundamentals/tensor_ops/README.md) |
| `layers/` | Linear、归一化、Dropout、Embedding、卷积、池化、残差、循环单元 | [LayerNorm](layers/layer_norm/README.md) |
| `attention/` | SDPA、MHA/GQA/MQA、位置编码、RoPE、FFN/SwiGLU、Transformer、KV Cache | [Attention](attention/scaled_dot_product/README.md) |
| `losses/` | CE/BCE、label smoothing、MSE/Huber、KL、InfoNCE | [分类损失](losses/classification/README.md) |
| `optimizers/` | SGD/Momentum、Adam/AdamW | [Adam 与 AdamW](optimizers/adam/README.md) |
| `training/` | 训练循环、梯度累积/裁剪、LR scheduler、AMP | [训练步骤](training/loop/README.md) |
| `rl/returns/` | discounted return、TD、GAE、n-step | [GAE](rl/returns/gae/README.md) |
| `rl/buffers/` | replay 与 rollout | [Rollout](rl/buffers/rollout/README.md) |
| `rl/policies/` | 离散、高斯、确定性策略与价值网络 | [Gaussian policy](rl/policies/gaussian/README.md) |
| `rl/objectives/` | policy gradient、PPO clipping | [PPO loss](rl/objectives/ppo/README.md) |
| `rl/updates/` | target 更新、参数冻结 | [梯度与同步](rl/updates/target/README.md) |
| `rl/algorithms/` | REINFORCE、Actor-Critic、DQN/Double/Dueling、PPO、DDPG、TD3、SAC | [DQN](rl/algorithms/dqn/README.md)、[PPO](rl/algorithms/ppo/README.md) |

每个学习单元的结构一致：

```text
attention/multi_head/
├── README.md          # 公式、shape、步骤、复杂度、边界、追问
├── implementation.py  # 核心代码与 __main__ 小例子
└── test_module.py     # 数值、梯度或行为测试
```

## 学习方式

1. 按 [ROADMAP.md](ROADMAP.md) 确认前置知识，阅读模块公式和 Tensor 约定。
2. 运行最小示例，跟踪每一步 shape；阅读测试，理解如何证明正确。
3. 到 [practice/](practice/README.md) 关闭答案手写，再完成 debug 题或组合题。
4. 在 [复习日志](practice/review_log.md) 记录耗时、错误和下次复习时间。

“仓库已实现”与“个人已掌握”分开记录。练习骨架中的 `NotImplementedError` 是有意留给你填写的空白；标准模块和示例均有完整实现。默认测试不收集练习目录。

## 端到端示例

```bash
# 合成数据训练：检查模块能组成可学习模型
python -m examples.supervised --model mlp --steps 100
python -m examples.supervised --model cnn --steps 100
python -m examples.supervised --model transformer --steps 100

# 离散控制；同一入口也支持 reinforce / actor_critic
python -m examples.cartpole --algorithm dqn --steps 5000 --seed 0
python -m examples.cartpole --algorithm ppo --steps 5000 --seed 0

# 连续控制；同一入口也支持 ddpg / td3
python -m examples.pendulum --algorithm sac --steps 3000 --seed 0

# CPU bfloat16 的 AMP 小例子
python -m training.mixed_precision.implementation
```

示例输出 JSON 指标。监督学习示例报告固定小训练集上的 loss/accuracy，旨在验证学习能力；不代表泛化性能。RL 的短训练用于检查环境、buffer 和更新逻辑，回报随种子与训练预算变化，不将某次回报作为算法正确性的证明。详细数据流见 [examples/README.md](examples/README.md)。

## 实现规范

- 所有函数参数与返回值带类型注释；Tensor 用 `torch.Tensor`/`Tensor` 表达类型，docstring 和关键语句注释补 shape、dtype、维度含义。
- 归约、广播、reshape、transpose、gather、matmul 旁标明关键形状。
- 手写当前模块的核心计算；官方实现只用作验证。组合单元允许显式复用基础构件。
- 核心计算与环境交互、数据生成、训练入口分离；不依赖全局配置或注册器。
- 独立模块通过输出、梯度、人工小例子或行为不变量验证；组合模型再做训练验证。
- 教学实现有明确支持范围，不追求覆盖 PyTorch 同名 API 的所有选项。优化器没有 checkpoint 序列化/参数组，RL algorithm 类当前按 CPU 示例组织，KV cache 是教学追加实现。

详细约定与后续新增模块模板见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 速查

- [Tensor 形状与类型约定](notes/tensor_conventions.md)
- [易错点与排错步骤](notes/pitfalls.md)
- [归一化、Attention、RL 算法对比](notes/comparisons.md)
- [RL 的边界与梯度流](notes/rl_dataflow.md)
- [官方参考资料](notes/references.md)

