# 手写：训练循环、梯度累积与裁剪

建议用时：20–35 分钟。先不看标准实现，允许查函数签名与 Tensor API。

## 任务

实现 [training__loop.py](../workspace/training__loop.py) 中的接口。保留类型标注，并在每次维度变化处补充 shape 注释。对算法组装单元，可以先完成 target/loss，再完成 update。

microbatch x:[Bi,...], y:[Bi,...]；损失 []；每个梯度与对应参数同形状。

## 允许使用的 API

允许标准库、PyTorch 张量操作、autograd，以及骨架中列出的已有模块。学习目标对应的核心计算必须自己展开，不能直接调用同名官方层/优化器/算法来代替。组合型模块允许复用已掌握的基础构件；官方实现只用于对照验证。

## 验收

- 用自己的话解释公式：逻辑 batch loss=Σ_i (Bi/ΣBj)*mean_loss_i；全局 L2 norm=sqrt(Σ_p ||grad_p||²)。
- 验证一个人工小例子，再验证关键梯度或行为；不能只检查 shape。
- 覆盖边界：不同大小 microbatch 不能简单平均各 loss。大 batch 等价要求逐样本可分解损失；BatchNorm、Dropout、InfoNCE 等会破坏条件。
- 用本地测试思路检查实现，并记录耗时、失败原因和修复过程。

## 自测方法

可把 [标准测试](../../training/loop/test_module.py) 复制为 workspace 中的 `test_training__loop.py`，将被测实现的 import 改为 `practice.workspace.training__loop`。保留其他测试辅助导入，然后在根目录执行：

```bash
python -m pytest practice/workspace/test_training__loop.py
```

默认全库 pytest 不收集 workspace，避免尚未完成的练习影响标准实现验证。

## 追问

为什么 optimizer.step 不能放进累积循环？梯度裁剪应在什么时候执行？

完成后再查看 [讲解](../../training/loop/README.md) 与 [标准答案](../../training/loop/implementation.py)。
