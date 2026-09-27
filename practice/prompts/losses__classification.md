# 手写：交叉熵、Label Smoothing、BCE with Logits

建议用时：20–35 分钟。先不看标准实现，允许查函数签名与 Tensor API。

## 任务

实现 [losses__classification.py](../workspace/losses__classification.py) 中的接口。保留类型标注，并在每次维度变化处补充 shape 注释。对算法组装单元，可以先完成 target/loss，再完成 update。

CE logits:[B,C]、target:int64 [B]；BCE logits/target 同形状 [...]；返回 scalar []。

## 允许使用的 API

允许标准库、PyTorch 张量操作、autograd，以及骨架中列出的已有模块。学习目标对应的核心计算必须自己展开，不能直接调用同名官方层/优化器/算法来代替。组合型模块允许复用已掌握的基础构件；官方实现只用于对照验证。

## 验收

- 用自己的话解释公式：CE=-log p_y；平滑目标为 (1-eps)*one_hot+eps/C；BCE=log(1+exp(z))-y*z。
- 验证一个人工小例子，再验证关键梯度或行为；不能只检查 shape。
- 覆盖边界：CE 输入是 logits，不能先 softmax。BCE 在 z=0 的梯度也需正确；只支持 mean reduction，无 ignore_index/class weights。
- 用本地测试思路检查实现，并记录耗时、失败原因和修复过程。

## 自测方法

可把 [标准测试](../../losses/classification/test_module.py) 复制为 workspace 中的 `test_losses__classification.py`，将被测实现的 import 改为 `practice.workspace.losses__classification`。保留其他测试辅助导入，然后在根目录执行：

```bash
python -m pytest practice/workspace/test_losses__classification.py
```

默认全库 pytest 不收集 workspace，避免尚未完成的练习影响标准实现验证。

## 追问

多标签分类为什么用 BCE？label smoothing 改变了什么目标？

完成后再查看 [讲解](../../losses/classification/README.md) 与 [标准答案](../../losses/classification/implementation.py)。
