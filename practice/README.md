# 手写、Debug 与综合练习

`prompts/` 为 49 个学习单元各提供一题；文件名以 `__` 对应原模块路径，例如 `attention__multi_head.md`。
`workspace/` 放置对应接口骨架；所有 `NotImplementedError` 都是有意的练习留白，不是标准实现未完成。

## 一轮练习

1. 只打开题目与骨架，先写公式、输入输出 shape、允许使用的基础模块。
2. 设定计时，完成实现与关键 shape 注释。
3. 复制该模块的测试到 workspace，并把被测模块 import 改到自己的实现。题目中有具体命令。
4. 若失败，先用小例子定位；完成后对照标准答案，解释差异。
5. 填写 review_log，并更新 ROADMAP 的个人掌握状态。

对于组合单元，允许复用已经掌握的模块；对于当前考察的核心算子，不能直接调用同名官方实现跳过计算。

## 进阶练习

- [Debug 题](debug_challenges.md)：shape 广播、梯度泄漏、mask、PPO 符号与 cache offset。
- [组合题](integration_challenges.md)：组装一个可训练小模型、DQN 更新、PPO rollout、SAC actor 更新。

默认 `pytest` 不收集 workspace；可用 `python -m pytest practice/workspace/test_题目名.py` 单独运行自己的练习测试。练习文件不会参与标准模块的类型检查。

