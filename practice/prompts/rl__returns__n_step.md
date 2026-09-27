# 手写：n-step Return

建议用时：30–50 分钟。先不看标准实现，允许查函数签名与 Tensor API。

## 任务

实现 [rl__returns__n_step.py](../workspace/rl__returns__n_step.py) 中的接口。保留类型标注，并在每次维度变化处补充 shape 注释。对算法组装单元，可以先完成 target/loss，再完成 update。

输入 [T,B] → target [T,B]；flags 为 bool。

## 允许使用的 API

允许标准库、PyTorch 张量操作、autograd，以及骨架中列出的已有模块。学习目标对应的核心计算必须自己展开，不能直接调用同名官方层/优化器/算法来代替。组合型模块允许复用已掌握的基础构件；官方实现只用于对照验证。

## 验收

- 用自己的话解释公式：G_t=Σ_(k=0..m-1) γ^k*r_(t+k)+γ^m*V(next)，m 不超过 n 或可用连续轨迹长度。
- 验证一个人工小例子，再验证关键梯度或行为；不能只检查 shape。
- 覆盖边界：样本必须按时间连续，不能直接把 replay 随机 batch 当成时间序列。截断与终止的 bootstrap 不同。
- 用本地测试思路检查实现，并记录耗时、失败原因和修复过程。

## 自测方法

可把 [标准测试](../../rl/returns/n_step/test_module.py) 复制为 workspace 中的 `test_rl__returns__n_step.py`，将被测实现的 import 改为 `practice.workspace.rl__returns__n_step`。保留其他测试辅助导入，然后在根目录执行：

```bash
python -m pytest practice/workspace/test_rl__returns__n_step.py
```

默认全库 pytest 不收集 workspace，避免尚未完成的练习影响标准实现验证。

## 追问

n=1 与单步 TD 如何对应？n 增大对偏差与方差有何影响？

完成后再查看 [讲解](../../rl/returns/n_step/README.md) 与 [标准答案](../../rl/returns/n_step/implementation.py)。
