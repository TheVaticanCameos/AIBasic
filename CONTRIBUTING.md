# 新增或修改学习单元

## 粒度

一个单元围绕一个可独立复习的概念。相关变体可放在同一目录，但类/函数应职责单一、可独立导入；例如 MHA/GQA 共用结构，RNN/LSTM/GRU 分别为独立 Cell。

需要复用时显式导入，不通过隐藏全局状态或服务定位器取依赖。基础函数尽量仅依赖 torch/标准库。组合算法可以依赖少量基础模块，README 列出相对链接。

## 类型和 Tensor 文档

```python
def select_action_values(q: Tensor, actions: Tensor) -> Tensor:
    """q floating [B,A]，actions int64 [B] -> 所选动作 Q [B]。"""
    selected = q.gather(1, actions[:, None])  # [B,1]
    return selected.squeeze(1)  # [B]
```

函数参数、返回值与关键状态有清晰类型；不要用 `Any` 代替可以明确的接口。标量返回用 `float`，保留梯度的标量用 `Tensor` 并注明 `[]`。不要只在变量名里写 shape，注释应解释维度含义。

mask 必须说明 True 的意义；RL flags 必须说明是否终止 bootstrap。可选 Tensor 明确为 `Tensor | None`；返回多个张量用固定长度 tuple 或具名 dataclass。

## 目录模板

```text
领域/知识点/
├── __init__.py
├── README.md
├── implementation.py
└── test_module.py
```

README 包含：用途/公式、输入输出、实现步骤、支持范围、易错点、复杂度、依赖、运行命令、测试依据、面试追问。implementation 提供小型 `__main__` 示例。

## 验证要求

选择与模块实际风险相符的测试，不为每行代码添加镜像测试：

- 可对照官方算子的层：对齐权重、dtype、eps、mask 语义后比较输出，关键模块同时比较梯度。
- 优化器：比较多次更新，包含缺失梯度与 weight decay。
- return/GAE/PPO：人工小例子包含正负优势、终止与截断。
- buffer：验证快照不随外部张量修改，历史数据不保留图。
- 组合算法：验证参数更新、detach 边界、目标网络和延迟更新行为。
- 训练示例：小数据学习或真实环境冒烟测试；随机回报不设脆弱的收敛阈值。

执行目标测试、Ruff、mypy；存在跨模块依赖变化时运行全量测试。修改已验证结果后更新 VALIDATION.md。不要修改 `practice/workspace` 中用户已经填写的内容来同步标准答案。

