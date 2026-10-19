# 手写：Discounted Return

建议用时：20–35 分钟。先不看标准实现，允许查函数签名与 Tensor API。

## 任务

实现 [rl__returns__discounted.py](../workspace/rl__returns__discounted.py) 中的接口。保留类型标注，并在每次维度变化处补充 shape 注释。对算法组装单元，可以先完成 target/loss，再完成 update。

rewards/next_values:[T,B]；terminated/truncated:bool [T,B]；输出 [T,B]。

## 允许使用的 API

允许标准库、PyTorch 张量操作、autograd，以及骨架中列出的已有模块。学习目标对应的核心计算必须自己展开，不能直接调用同名官方层/优化器/算法来代替。组合型模块允许复用已掌握的基础构件；官方实现只用于对照验证。

## 验收

- 用自己的话解释公式：G_t=r_t+γ*G_(t+1)；真正终止后续为 0，截断处替换为 V(next_obs)。
- 验证一个人工小例子，再验证关键梯度或行为；不能只检查 shape。
- 覆盖边界：next_values 必须来自 reset 前下一观测；Monte Carlo 不带价值时传全零。terminated 与 truncated 同时为真时终止优先。
- 用本地测试思路检查实现，并记录耗时、失败原因和修复过程。

## 自测方法

无需复制测试文件。在仓库根目录执行，自动复用对应标准测试验证你的实现：

```bash
.venv/bin/python -m practice.check rl__returns__discounted
```

激活 `.venv` 后可简写为 `python -m practice.check rl__returns__discounted`。
用 `python -m practice.check --all` 批量测试无占位语句的练习，或用 `--list` 查看状态。
测试失败会显示断言差异和你的实现位置。入口不会修改练习或标准答案。
更多参数与部分完成时的行为见 [练习说明](../README.md)。

## 追问

采集步数耗尽与 MDP 真正终止有什么区别？return 与 reward 有何区别？

完成后再查看 [讲解](../../rl/returns/discounted/README.md) 与 [标准答案](../../rl/returns/discounted/implementation.py)。
