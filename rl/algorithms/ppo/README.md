# PPO 算法组装

优先级：**P0**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

GAE → 固定优势/return → 多轮 clipped policy loss + value_coef*MSE - entropy_coef*entropy。

## Tensor 约定

rollout:[T,B,...] 展平为 N=T*B；minibatch:[M,...]。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

采样时保存旧统计量；更新前全 rollout 标准化优势；每 epoch 打乱索引、minibatch 优化、裁剪梯度。

## 易错点与支持边界

不重算 old_log_probs，不在各 epoch 更新 advantage。使用独立 actor/value 网络；未实现 value clipping 与 KL early stop。

## 复杂度

epochs*每轮数据量的 actor/value 前后向，rollout 内存 O(TB(O+1))。

## 接口与依赖

`act(self, observations: Tensor) -> tuple[Tensor, Tensor, Tensor]`

obs [B,O] -> action/log_prob/value，各 [B]；固定采样时策略统计量。

`update(self, rollout: RolloutBatch) -> dict[str, float]`

rollout [T,B,...] -> 标量日志；展平为 N=T*B，旧数据始终 detach。

仓库内依赖：[rl.buffers.rollout.implementation](../../buffers/rollout/implementation.py)、[rl.objectives.ppo.implementation](../../objectives/ppo/implementation.py)、[rl.policies.categorical.implementation](../../policies/categorical/implementation.py)、[rl.returns.gae.implementation](../../returns/gae/implementation.py)、[rl.returns.td.implementation](../../returns/td/implementation.py)

## 运行与验证

在仓库根目录运行：

```bash
python -m rl.algorithms.ppo.implementation
python -m pytest rl/algorithms/ppo/test_module.py
```

测试关注：

- `test_multiepoch_update_preserves_old_data_and_updates_actor_value`

## 面试追问与练习

为什么同一 rollout 能更新多轮但不能无限 replay？rollout 尾部如何 bootstrap？

[独立手写题与起始接口](../../../practice/prompts/rl__algorithms__ppo.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
