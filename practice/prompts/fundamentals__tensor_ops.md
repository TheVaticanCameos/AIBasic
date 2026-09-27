# 手写：张量操作与形状变换

建议用时：20–35 分钟。先不看标准实现，允许查函数签名与 Tensor API。

## 任务

实现 [fundamentals__tensor_ops.py](../workspace/fundamentals__tensor_ops.py) 中的接口。保留类型标注，并在每次维度变化处补充 shape 注释。对算法组装单元，可以先完成 target/loss，再完成 update。

[B,T,D] ↔ [B,H,T,d]，D=H*d；Q:[B,A] + actions:[B] → [B]。

## 允许使用的 API

允许标准库、PyTorch 张量操作、autograd，以及骨架中列出的已有模块。学习目标对应的核心计算必须自己展开，不能直接调用同名官方层/优化器/算法来代替。组合型模块允许复用已掌握的基础构件；官方实现只用于对照验证。

## 验收

- 用自己的话解释公式：reshape/transpose 改变维度组织；gather 按样本挑选动作；einsum 显式表达求和维度。
- 验证一个人工小例子，再验证关键梯度或行为；不能只检查 shape。
- 覆盖边界：transpose 后 tensor 可能不连续；view 不一定合法，reshape 必要时复制。gather 的索引为 int64。
- 用本地测试思路检查实现，并记录耗时、失败原因和修复过程。

## 自测方法

可把 [标准测试](../../fundamentals/tensor_ops/test_module.py) 复制为 workspace 中的 `test_fundamentals__tensor_ops.py`，将被测实现的 import 改为 `practice.workspace.fundamentals__tensor_ops`。保留其他测试辅助导入，然后在根目录执行：

```bash
python -m pytest practice/workspace/test_fundamentals__tensor_ops.py
```

默认全库 pytest 不收集 workspace，避免尚未完成的练习影响标准实现验证。

## 追问

为什么 transpose 后不能总用 view？gather 与高级索引的梯度有什么特点？

完成后再查看 [讲解](../../fundamentals/tensor_ops/README.md) 与 [标准答案](../../fundamentals/tensor_ops/implementation.py)。
