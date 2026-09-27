# Rollout Buffer

优先级：**P0**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

按时间堆叠同一旧策略采集的 transition；finish 消费并清空。

## Tensor 约定

add 单步 [B,...]；finish 返回 [T,B,...]；action 为离散 [T,B]。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

每次 detach+clone 保存 old_log_prob/value/next_value；按字段 stack。

## 易错点与支持边界

RolloutBatch 在 add 时省略 T 维。当前实现用于离散 PPO；不能像 replay 那样无限复用旧数据。

## 复杂度

追加按数据量复制；finish O(TB(O+1))，内存同阶。

## 接口与依赖

`add(self, step: RolloutBatch) -> None`

单步字段省略 T 维，各 [B,...]；detach + clone 固定旧策略数据。

`finish(self) -> RolloutBatch`

按时间堆叠为 [T,B,...] 并清空，防止下一轮误用旧 rollout。

仓库内依赖：无；可单独复制 implementation.py，仅依赖标准库/PyTorch。

## 运行与验证

在仓库根目录运行：

```bash
python -m rl.buffers.rollout.implementation
python -m pytest rl/buffers/rollout/test_module.py
```

测试关注：

- `test_snapshot_is_detached_and_finish_clears`

## 面试追问与练习

old log-prob 为什么必须固定？reset 前下一观测的 value 应在何时计算？

[独立手写题与起始接口](../../../practice/prompts/rl__buffers__rollout.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
