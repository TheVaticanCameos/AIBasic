# 确定性策略与 Continuous Q

优先级：**P1**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

a=actor(s)；Q=critic(concat(s,a))。

## Tensor 约定

obs:[B,O]，action:[B,A] → Q:[B]；actor 输出经 tanh 限制。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

actor 与 critic 使用独立 MLP；critic 拼接状态和动作，以便梯度传回 actor 的动作。

## 易错点与支持边界

actor 更新时冻结 critic 参数但不能 no_grad 整个 critic 前向，否则动作梯度也断开。

## 复杂度

MLP 主开销 O(B*hidden²)，输入层分别依赖 O 与 O+A。

## 接口与依赖

`forward(self, observations: Tensor) -> Tensor`

observations [B,O] -> actions [B,A]，范围 [-1,1]。

`forward(self, observations: Tensor, actions: Tensor) -> Tensor`

obs [B,O], actions [B,A] -> Q [B]。

仓库内依赖：无；可单独复制 implementation.py，仅依赖标准库/PyTorch。

## 运行与验证

在仓库根目录运行：

```bash
python -m rl.policies.continuous.implementation
python -m pytest rl/policies/continuous/test_module.py
```

测试关注：

- `test_actor_bounds_and_action_to_value_gradient`

## 面试追问与练习

为什么 DDPG 不需要 log-prob？Q 对 action 的导数如何训练 actor？

[独立手写题与起始接口](../../../practice/prompts/rl__policies__continuous.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
