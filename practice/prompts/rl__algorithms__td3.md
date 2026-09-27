# 手写：TD3

建议用时：30–50 分钟。先不看标准实现，允许查函数签名与 Tensor API。

## 任务

实现 [rl__algorithms__td3.py](../workspace/rl__algorithms__td3.py) 中的接口。保留类型标注，并在每次维度变化处补充 shape 注释。对算法组装单元，可以先完成 target/loss，再完成 update。

obs:[B,O]，actions/noise:[B,A]，双 Q:[B]。

## 允许使用的 API

允许标准库、PyTorch 张量操作、autograd，以及骨架中列出的已有模块。学习目标对应的核心计算必须自己展开，不能直接调用同名官方层/优化器/算法来代替。组合型模块允许复用已掌握的基础构件；官方实现只用于对照验证。

## 验收

- 用自己的话解释公式：target=r+γ min(Q1target,Q2target)(next,clip(actor_target(next)+noise))。
- 验证一个人工小例子，再验证关键梯度或行为；不能只检查 shape。
- 覆盖边界：目标平滑噪声与行为探索噪声用途不同。actor 仅使用 Q1；延迟步日志 actor_loss=None 表示本轮未更新。
- 用本地测试思路检查实现，并记录耗时、失败原因和修复过程。

## 自测方法

可把 [标准测试](../../rl/algorithms/td3/test_module.py) 复制为 workspace 中的 `test_rl__algorithms__td3.py`，将被测实现的 import 改为 `practice.workspace.rl__algorithms__td3`。保留其他测试辅助导入，然后在根目录执行：

```bash
python -m pytest practice/workspace/test_rl__algorithms__td3.py
```

默认全库 pytest 不收集 workspace，避免尚未完成的练习影响标准实现验证。

## 追问

TD3 的三个改进分别针对什么问题？为何取两个 critic 的最小值？

完成后再查看 [讲解](../../rl/algorithms/td3/README.md) 与 [标准答案](../../rl/algorithms/td3/implementation.py)。
