# 手写：MSE 与 Huber

建议用时：20–35 分钟。先不看标准实现，允许查函数签名与 Tensor API。

## 任务

实现 [losses__regression.py](../workspace/losses__regression.py) 中的接口。保留类型标注，并在每次维度变化处补充 shape 注释。对算法组装单元，可以先完成 target/loss，再完成 update。

prediction/target 必须同形状 [...]；输出 scalar []。

## 允许使用的 API

允许标准库、PyTorch 张量操作、autograd，以及骨架中列出的已有模块。学习目标对应的核心计算必须自己展开，不能直接调用同名官方层/优化器/算法来代替。组合型模块允许复用已掌握的基础构件；官方实现只用于对照验证。

## 验收

- 用自己的话解释公式：MSE=mean((pred-target)²)；Huber 小误差为 0.5e²，大误差为 δ*(|e|-0.5δ)。
- 验证一个人工小例子，再验证关键梯度或行为；不能只检查 shape。
- 覆盖边界：[B,1] 与 [B] 相减会广播为 [B,B]，所以主动拒绝。Huber 与 SmoothL1 相差 delta 缩放。
- 用本地测试思路检查实现，并记录耗时、失败原因和修复过程。

## 自测方法

可把 [标准测试](../../losses/regression/test_module.py) 复制为 workspace 中的 `test_losses__regression.py`，将被测实现的 import 改为 `practice.workspace.losses__regression`。保留其他测试辅助导入，然后在根目录执行：

```bash
python -m pytest practice/workspace/test_losses__regression.py
```

默认全库 pytest 不收集 workspace，避免尚未完成的练习影响标准实现验证。

## 追问

Huber 为什么对异常值更稳健？MSE 的 mean 究竟归约哪些维度？

完成后再查看 [讲解](../../losses/regression/README.md) 与 [标准答案](../../losses/regression/implementation.py)。
