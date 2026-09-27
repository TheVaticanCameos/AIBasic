# DDPG

优先级：**P1**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

critic 拟合 r+γQtarget(next,actor_target(next))；actor 最小化 -Q(s,actor(s))。

## Tensor 约定

obs:[B,O]，标准化 actions:[B,A]，Q/target:[B]。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

更新 critic；冻结 critic 参数但保留动作梯度更新 actor；软更新两个 target。

## 易错点与支持边界

探索噪声由 act 参数控制，评估设为 0。动作都在 [-1,1]，环境缩放在 examples 边界完成。

## 复杂度

每步一个 critic 和 actor 更新，加目标网络前向。

## 接口与依赖

`act(self, observations: Tensor, noise_std: float=0.0) -> Tensor`

obs [B,O] -> 标准化动作 [B,A]；评估时 noise_std=0。

`update(self, batch: TransitionBatch) -> dict[str, float]`

obs [B,O], actions [B,A]；动作已经映射到 [-1,1] 空间。

仓库内依赖：[rl.buffers.replay.implementation](../../buffers/replay/implementation.py)、[rl.policies.continuous.implementation](../../policies/continuous/implementation.py)、[rl.returns.td.implementation](../../returns/td/implementation.py)、[rl.updates.target.implementation](../../updates/target/implementation.py)

## 运行与验证

在仓库根目录运行：

```bash
python -m rl.algorithms.ddpg.implementation
python -m pytest rl/algorithms/ddpg/test_module.py
```

测试关注：

- `test_actor_updates_critic_frozen_during_actor_and_targets_detached`

## 面试追问与练习

确定性策略梯度通过哪些路径传播？DDPG 的探索来自哪里？

[独立手写题与起始接口](../../../practice/prompts/rl__algorithms__ddpg.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
