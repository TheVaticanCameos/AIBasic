# Replay Buffer

优先级：**P0**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

环形写指针 position=(position+1)%capacity；随机有放回采样。

## Tensor 约定

存储 obs:[C,O]、action:[C] 或 [C,A]、其他标量:[C]；采样首维变为 B。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

预分配 CPU 张量、逐条复制、记录真实 next_obs 与两个 flags、按随机索引取 batch。

## 易错点与支持边界

写入必须复制并 detach，避免外部环境复用数组导致数据变动。不保存计算图；不做优先级采样。

## 复杂度

单条写入 O(O+A)，采样 O(B*(O+A))，存储 O(C*(O+A))。

## 接口与依赖

`add(self, observation: Tensor, action: Tensor, reward: float, next_observation: Tensor, terminated: bool, truncated: bool) -> None`

单条数据：obs/next_obs [O]，action [] 或 [A]；立即复制以免外部复用覆盖。

`sample(self, batch_size: int, generator: torch.Generator | None=None) -> TransitionBatch`

有放回采样，返回 B=batch_size；generator 用于复现。

仓库内依赖：无；可单独复制 implementation.py，仅依赖标准库/PyTorch。

## 运行与验证

在仓库根目录运行：

```bash
python -m rl.buffers.replay.implementation
python -m pytest rl/buffers/replay/test_module.py
```

测试关注：

- `test_overwrite_copy_sampling_and_terminal_flags`

## 面试追问与练习

为什么 off-policy 方法能使用 replay？环形覆盖后哪些索引仍有效？

[独立手写题与起始接口](../../../practice/prompts/rl__buffers__replay.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
