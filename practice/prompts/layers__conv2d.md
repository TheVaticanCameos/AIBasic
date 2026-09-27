# 手写：Conv2d 循环与 unfold 实现

建议用时：20–35 分钟。先不看标准实现，允许查函数签名与 Tensor API。

## 任务

实现 [layers__conv2d.py](../workspace/layers__conv2d.py) 中的接口。保留类型标注，并在每次维度变化处补充 shape 注释。对算法组装单元，可以先完成 target/loss，再完成 update。

X:[B,C,H,W]，W:[O,C,K,K] → [B,O,Hout,Wout]；Hout=floor((H+2P-K)/S)+1。

## 允许使用的 API

允许标准库、PyTorch 张量操作、autograd，以及骨架中列出的已有模块。学习目标对应的核心计算必须自己展开，不能直接调用同名官方层/优化器/算法来代替。组合型模块允许复用已掌握的基础构件；官方实现只用于对照验证。

## 验收

- 用自己的话解释公式：Y[b,o,i,j]=Σ(c,u,v) Xpad[b,c,i*S+u,j*S+v]*W[o,c,u,v]+b[o]。
- 验证一个人工小例子，再验证关键梯度或行为；不能只检查 shape。
- 覆盖边界：实际计算为互相关，不翻转核。限定方形核、整数 stride/padding、groups=1、dilation=1、合法输出尺寸。
- 用本地测试思路检查实现，并记录耗时、失败原因和修复过程。

## 自测方法

可把 [标准测试](../../layers/conv2d/test_module.py) 复制为 workspace 中的 `test_layers__conv2d.py`，将被测实现的 import 改为 `practice.workspace.layers__conv2d`。保留其他测试辅助导入，然后在根目录执行：

```bash
python -m pytest practice/workspace/test_layers__conv2d.py
```

默认全库 pytest 不收集 workspace，避免尚未完成的练习影响标准实现验证。

## 追问

stride/padding 如何影响输出尺寸？unfold 为什么可能更耗内存？

完成后再查看 [讲解](../../layers/conv2d/README.md) 与 [标准答案](../../layers/conv2d/implementation.py)。
