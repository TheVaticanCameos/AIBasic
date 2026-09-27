# Categorical Policy 与 Value Network

优先级：**P0**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

π(a|s)=softmax(logits(s))；V(s) 为单标量。

## Tensor 约定

obs:[B,O] → logits:[B,A]；sample/log_prob/entropy/value 均 [B]。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

策略输出 logits 并交给 Categorical；价值网络输出 [B,1] 后仅 squeeze 最后一维。

## 易错点与支持边界

不要 squeeze() 删除 B=1 的 batch 轴；离散采样不可直接重参数，用 log-prob 梯度。

## 复杂度

MLP 时间随隐藏维度增长，最后投影 O(BHA)。

## 接口与依赖

`forward(self, observations: Tensor) -> Tensor`

observations [B,O] -> logits [B,A]，不提前 softmax。

`distribution(self, observations: Tensor) -> Categorical`

obs [B,O] -> batch_shape=[B], event_shape=[] 的分布。

`forward(self, observations: Tensor) -> Tensor`

observations [B,O] -> values [B]，仅 squeeze 最后一维。

仓库内依赖：无；可单独复制 implementation.py，仅依赖标准库/PyTorch。

## 运行与验证

在仓库根目录运行：

```bash
python -m rl.policies.categorical.implementation
python -m pytest rl/policies/categorical/test_module.py
```

测试关注：

- `test_distribution_log_prob_entropy_and_singleton_batch`

## 面试追问与练习

为什么策略网络输出 logits 更稳健？entropy 奖励鼓励什么行为？

[独立手写题与起始接口](../../../practice/prompts/rl__policies__categorical.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
