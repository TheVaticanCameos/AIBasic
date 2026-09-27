# 手写：推理 KV Cache

建议用时：30–50 分钟。先不看标准实现，允许查函数签名与 Tensor API。

## 任务

实现 [attention__kv_cache.py](../workspace/attention__kv_cache.py) 中的接口。保留类型标注，并在每次维度变化处补充 shape 注释。对算法组装单元，可以先完成 target/loss，再完成 update。

新输入 [B,Tnew,D]；缓存 [B,Hkv,Tpast,d]；mask [Tnew,Tpast+Tnew]。

## 允许使用的 API

允许标准库、PyTorch 张量操作、autograd，以及骨架中列出的已有模块。学习目标对应的核心计算必须自己展开，不能直接调用同名官方层/优化器/算法来代替。组合型模块允许复用已掌握的基础构件；官方实现只用于对照验证。

## 验收

- 用自己的话解释公式：Kcache=concat(Kpast,Knew)，Vcache 同理；新 Q 关注所有合法过去 key。
- 验证一个人工小例子，再验证关键梯度或行为；不能只检查 shape。
- 覆盖边界：只允许 eval + no_grad。换序列须 clear；batch 顺序不能变化。不集成 RoPE 或 padding mask，输入应已带绝对位置编码。
- 用本地测试思路检查实现，并记录耗时、失败原因和修复过程。

## 自测方法

可把 [标准测试](../../attention/kv_cache/test_module.py) 复制为 workspace 中的 `test_attention__kv_cache.py`，将被测实现的 import 改为 `practice.workspace.attention__kv_cache`。保留其他测试辅助导入，然后在根目录执行：

```bash
python -m pytest practice/workspace/test_attention__kv_cache.py
```

默认全库 pytest 不收集 workspace，避免尚未完成的练习影响标准实现验证。

## 追问

为什么只缓存 KV？生成多个 token 的 chunk 为什么不能简单使用左上角三角 mask？

完成后再查看 [讲解](../../attention/kv_cache/README.md) 与 [标准答案](../../attention/kv_cache/implementation.py)。
