# 手写：Gaussian Policy 与 Tanh 修正

建议用时：30–50 分钟。先不看标准实现，允许查函数签名与 Tensor API。

## 任务

实现 [rl__policies__gaussian.py](../workspace/rl__policies__gaussian.py) 中的接口。保留类型标注，并在每次维度变化处补充 shape 注释。对算法组装单元，可以先完成 target/loss，再完成 update。

obs:[B,O] → mean/log_std/actions:[B,A]；log_prob 对动作维求和后 [B]。

## 允许使用的 API

允许标准库、PyTorch 张量操作、autograd，以及骨架中列出的已有模块。学习目标对应的核心计算必须自己展开，不能直接调用同名官方层/优化器/算法来代替。组合型模块允许复用已掌握的基础构件；官方实现只用于对照验证。

## 验收

- 用自己的话解释公式：u=μ+σϵ，a=tanh(u)；log π(a)=Σ(log N(u)-log(1-tanh²u))。
- 验证一个人工小例子，再验证关键梯度或行为；不能只检查 shape。
- 覆盖边界：log_prob 不能对动作维取 mean。输出密度定义在标准 [-1,1] 空间；若计算环境尺度密度还要减 log scale 的和。
- 用本地测试思路检查实现，并记录耗时、失败原因和修复过程。

## 自测方法

可把 [标准测试](../../rl/policies/gaussian/test_module.py) 复制为 workspace 中的 `test_rl__policies__gaussian.py`，将被测实现的 import 改为 `practice.workspace.rl__policies__gaussian`。保留其他测试辅助导入，然后在根目录执行：

```bash
python -m pytest practice/workspace/test_rl__policies__gaussian.py
```

默认全库 pytest 不收集 workspace，避免尚未完成的练习影响标准实现验证。

## 追问

sample 与 rsample 有何不同？tanh 为什么改变概率密度？

完成后再查看 [讲解](../../rl/policies/gaussian/README.md) 与 [标准答案](../../rl/policies/gaussian/implementation.py)。
