# 手写：稳定 softmax / log-softmax / logsumexp

建议用时：20–35 分钟。先不看标准实现，允许查函数签名与 Tensor API。

## 任务

实现 [fundamentals__stable_ops.py](../workspace/fundamentals__stable_ops.py) 中的接口。保留类型标注，并在每次维度变化处补充 shape 注释。对算法组装单元，可以先完成 target/loss，再完成 update。

输入 floating [...,D,...]；沿 dim 归约，keepdim 决定是否保留单例轴。

## 允许使用的 API

允许标准库、PyTorch 张量操作、autograd，以及骨架中列出的已有模块。学习目标对应的核心计算必须自己展开，不能直接调用同名官方层/优化器/算法来代替。组合型模块允许复用已掌握的基础构件；官方实现只用于对照验证。

## 验收

- 用自己的话解释公式：logsumexp(x)=m+log Σ exp(x-m)；log_softmax=x-logsumexp(x)。
- 验证一个人工小例子，再验证关键梯度或行为；不能只检查 shape。
- 覆盖边界：先 softmax 再 log 会丢失小概率精度。此单元要求有限 logits；全 -inf mask 由 Attention 单元专门处理。
- 用本地测试思路检查实现，并记录耗时、失败原因和修复过程。

## 自测方法

可把 [标准测试](../../fundamentals/stable_ops/test_module.py) 复制为 workspace 中的 `test_fundamentals__stable_ops.py`，将被测实现的 import 改为 `practice.workspace.fundamentals__stable_ops`。保留其他测试辅助导入，然后在根目录执行：

```bash
python -m pytest practice/workspace/test_fundamentals__stable_ops.py
```

默认全库 pytest 不收集 workspace，避免尚未完成的练习影响标准实现验证。

## 追问

为什么平移不改变 softmax？为什么交叉熵应直接从 logits 计算？

完成后再查看 [讲解](../../fundamentals/stable_ops/README.md) 与 [标准答案](../../fundamentals/stable_ops/implementation.py)。
