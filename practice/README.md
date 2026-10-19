# 手写、Debug 与综合练习

`prompts/` 为 49 个学习单元各提供一题；文件名以 `__` 对应原模块路径，例如 `attention__multi_head.md`。
`workspace/` 放置对应接口骨架；所有 `NotImplementedError` 都是有意的练习留白，不是标准实现未完成。

## 一键验证

在仓库根目录执行，无需复制或修改测试文件：

```bash
# 测试指定练习；名称就是文件名去掉 .py
.venv/bin/python -m practice.check fundamentals__tensor_ops

# 也可以直接传入练习文件路径
.venv/bin/python -m practice.check practice/workspace/fundamentals__tensor_ops.py

# 一次测试所有无占位语句的练习
.venv/bin/python -m practice.check --all

# 查看练习状态，不执行练习代码
.venv/bin/python -m practice.check --list

# 需要时，-- 后传入 pytest 参数：失败即停、详细输出、筛选用例
.venv/bin/python -m practice.check fundamentals__tensor_ops -- -x -vv
.venv/bin/python -m practice.check fundamentals__tensor_ops -- -k permute
```

激活 `.venv` 后，可以将 `.venv/bin/python` 简写为 `python`。需已安装开发依赖：`python -m pip install -e '.[dev]'`。

入口会自动找到对应 `test_module.py`，仅把被测模块的导入改成你的 workspace 实现，在临时目录运行后自动清理。断言、参数化与根目录 fixture 继续复用；标准答案、练习文件和已有的手动测试文件都不会被修改。失败 traceback 会指向你的实现；测试本身的路径显示为 `.practice-tests-.../test_练习名.py`。

组合模块继续使用其显式导入的依赖，入口不会把其他未完成练习注入这些依赖。缺少函数、语法错误和运行错误都会报失败，不会回退到标准答案。

`--all` 用 AST 检查实际的 `raise NotImplementedError(...)` 占位语句，忽略注释中的同名文本。它跳过有占位语句或缺少文件的练习，语法错误仍会报错。“可测试”仅代表未发现占位，不代表实现正确；也不会识别所有形式的未完成代码（例如自己写的 `pass`）。没有任何可测试练习时返回非零状态，不会显示全部通过。明确指定某个练习时，即使尚有占位也会运行测试，方便边写边验证。

## 一轮练习

1. 只打开题目与骨架，先写公式、输入输出 shape、允许使用的基础模块。
2. 设定计时，完成实现与关键 shape 注释。
3. 执行 `python -m practice.check 练习名`，自动复用对应标准测试验证自己的实现。
4. 若失败，先用小例子定位；完成后对照标准答案，解释差异。
5. 填写 review_log，并更新 ROADMAP 的个人掌握状态。

对于组合单元，允许复用已经掌握的模块；对于当前考察的核心算子，不能直接调用同名官方实现跳过计算。

## 进阶练习

- [Debug 题](debug_challenges.md)：shape 广播、梯度泄漏、mask、PPO 符号与 cache offset。
- [组合题](integration_challenges.md)：组装一个可训练小模型、DQN 更新、PPO rollout、SAC actor 更新。

默认根目录 `pytest` 仍验证标准实现，不收集 workspace。已有的手动测试脚本可以保留，或用于增加自己的测试；它们不会被 `practice.check` 重复收集。练习文件不会参与标准模块的类型检查。

