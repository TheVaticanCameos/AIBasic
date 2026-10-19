# 验证记录

验证日期：2026-09-27。所有结果来自本地实际执行。

## 环境

| 项目 | 实际版本 |
|---|---|
| Python | 3.13.5 |
| PyTorch | 2.14.0+cpu |
| NumPy | 2.3.3 |
| Gymnasium | 1.3.0 |
| pytest | 9.1.1 |
| Ruff | 0.16.9 |
| mypy | 2.3.1 |

使用本仓库 `.venv`，复用系统 CPU PyTorch，已执行 `pip install -e '.[dev,rl]'`。CUDA 不可用，未验证 CUDA/float16 分支；AMP 实际验证 CPU bfloat16。声明的最低依赖版本未逐版本矩阵测试。

## 代码检查

```text
python -m pytest
81 passed in 3.74s

ruff check .
All checks passed!

mypy fundamentals layers attention losses optimizers training rl examples
Success: no issues found in 115 source files
```

全量测试包含 49 个学习单元的本地测试，以及 3 个监督学习和 7 个 RL 算法的训练/交互集成测试。Gymnasium 已安装，本次没有因缺少环境依赖而跳过 RL 集成测试。

另外逐一执行了 49 个 `implementation.py` 的 `__main__` 最小例子，全部成功。每个学习单元均具备 README、implementation、test_module 和练习题/接口骨架。

## 关键证据

| 需求 | 验证依据 |
|---|---|
| 数值稳定性 | 极大 logits、float32 巨大公共偏移、BCE 在零点的梯度、Attention 全 mask 行 |
| 输出与梯度一致 | Linear、归一化、卷积双实现、池化、循环单元、SDPA、分类/回归/KL 损失与 PyTorch 对照 |
| 自动求导 | 自定义 square 的 gradcheck 与 gradgradcheck |
| Attention 行为 | 因果不泄漏、GQA/MQA head 映射、RoPE 相对位置、KV 分块与全序列一致 |
| 优化器与训练 | 多次 SGD/Adam/AdamW 更新、缺失梯度、不等长 microbatch、全局梯度裁剪、CPU AMP |
| RL 边界 | termination/truncation/rollout 尾部、手算 GAE/return/n-step、buffer 快照与 detach |
| RL 更新 | PPO 正负优势 clipping、旧策略不变、actor/value 更新、Double DQN 选动作/估值分离、TD3 延迟更新、SAC 最小双 Q 与熵项符号 |
| 模块化与类型 | 独立运行示例、显式依赖链接、Ruff 注解规则、mypy 检查 |

相关测试在各模块 README 中列出。shape/dtype/语义由接口文档和关键变换注释表达，Python 的 `Tensor` 类型本身不静态验证维度。

## 实际命令行训练

| 命令 | 结果摘要 |
|---|---|
| `python -m examples.supervised --model transformer --steps 100` | loss 2.147824 → 0.000008919，训练集 accuracy 1.0 |
| `python -m examples.cartpole --algorithm dqn --steps 1000 --seed 0` | 937 次更新，78 个完整 episode，独立评估平均回报 13.67 |
| `python -m examples.cartpole --algorithm ppo --steps 1024 --seed 0` | 8 批 rollout 更新，38 个完整 episode，独立评估平均回报 27.0 |
| `python -m examples.pendulum --algorithm sac --steps 300 --seed 0` | 173 次更新，1 个完整 episode，回报 -1125.92 |

这些短 RL 运行只证明采样和更新流程可执行，**未证明收敛或达到环境得分标准**。监督学习指标是固定合成训练集上的拟合结果，不是泛化评估。REINFORCE、Actor-Critic、DDPG、TD3 的环境运行由集成测试覆盖。

## 范围

已实现方案中的 P0/P1 教学核心及复习材料。P2 的 RLHF/扩散/GNN/分布式/kernel 实现未纳入本次范围，记录于 ROADMAP 的扩展方向。各模块主动简化的选项、设备限制和复杂度均写在对应 README 中。

## 2026-10-19：练习自动测试入口

新增 `python -m practice.check 练习名`，以及文件路径、`--all`、`--list` 和 `--` 后转交 pytest 参数的入口。49 道练习说明均改用该入口，无需复制测试脚本。

- 自动测试入口的 11 个回归用例通过：覆盖正确/错误/缺少接口的学生实现、fixture 与参数化保留、标准答案不被误测、占位识别、临时文件清理、批量与路径入口。
- 为 tensor_ops 增加 head 数量为 0、负数、不可整除的 3 个参数化边界用例。
- 标准实现与测试工具：全量 `pytest` **95 passed in 4.62s**；Ruff 通过；mypy（包含 `practice/check.py`）检查 116 个源文件通过。
- 实际指定当前用户的 tensor_ops 练习执行，正确报告 `num_heads=0` 时先做取模导致的 `ZeroDivisionError`。用户练习代码未修改；默认全库标准测试通过不代表个人练习通过。
- `--list` 发现 49 个练习；当时 1 个无占位语句，48 个待完成。`--all` 的自动筛选已实际执行验证。
