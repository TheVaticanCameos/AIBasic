# 手写：Adam 与 AdamW

建议用时：20–35 分钟。先不看标准实现，允许查函数签名与 Tensor API。

## 任务

实现 [optimizers__adam.py](../workspace/optimizers__adam.py) 中的接口。保留类型标注，并在每次维度变化处补充 shape 注释。对算法组装单元，可以先完成 target/loss，再完成 update。

每参数状态 first/second 与参数同形状 [...]；step 为每参数独立整数。

## 允许使用的 API

允许标准库、PyTorch 张量操作、autograd，以及骨架中列出的已有模块。学习目标对应的核心计算必须自己展开，不能直接调用同名官方层/优化器/算法来代替。组合型模块允许复用已掌握的基础构件；官方实现只用于对照验证。

## 验收

- 用自己的话解释公式：m=β1*m+(1-β1)g；v=β2*v+(1-β2)g²；用 m/(1-β1^t)、v/(1-β2^t) 更新。
- 验证一个人工小例子，再验证关键梯度或行为；不能只检查 shape。
- 覆盖边界：Adam 的 L2 项加入梯度；AdamW 解耦衰减。grad=None 不递增该参数 step；不实现 AMSGrad/param groups/序列化。
- 用本地测试思路检查实现，并记录耗时、失败原因和修复过程。

## 自测方法

可把 [标准测试](../../optimizers/adam/test_module.py) 复制为 workspace 中的 `test_optimizers__adam.py`，将被测实现的 import 改为 `practice.workspace.optimizers__adam`。保留其他测试辅助导入，然后在根目录执行：

```bash
python -m pytest practice/workspace/test_optimizers__adam.py
```

默认全库 pytest 不收集 workspace，避免尚未完成的练习影响标准实现验证。

## 追问

为什么需要偏差修正？Adam 加 L2 与 AdamW 为何不等价？

完成后再查看 [讲解](../../optimizers/adam/README.md) 与 [标准答案](../../optimizers/adam/implementation.py)。
