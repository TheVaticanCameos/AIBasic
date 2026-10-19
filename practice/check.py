"""复用标准测试验证练习：python -m practice.check fundamentals__tensor_ops。"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("fundamentals", "layers", "attention", "losses", "optimizers", "training", "rl")
Status = Literal["ready", "unfinished", "syntax_error", "missing"]
STATUS_LABELS: dict[Status, str] = {
    "ready": "可测试（无占位语句）",
    "unfinished": "待完成",
    "syntax_error": "语法错误",
    "missing": "缺少练习文件",
}


@dataclass(frozen=True)
class Exercise:
    name: str
    source_module: str
    workspace: Path
    standard_test: Path

    @property
    def practice_module(self) -> str:
        return f"practice.workspace.{self.name}"


def discover(root: Path) -> dict[str, Exercise]:
    """按标准学习目录建立索引，不收集用户手动复制的 test_*.py。"""
    exercises: dict[str, Exercise] = {}
    for domain in DOMAINS:
        for test in sorted((root / domain).rglob("test_module.py")):
            parts = test.parent.relative_to(root).parts
            name = "__".join(parts)
            exercises[name] = Exercise(
                name=name,
                source_module=".".join((*parts, "implementation")),
                workspace=root / "practice" / "workspace" / f"{name}.py",
                standard_test=test,
            )
    return dict(sorted(exercises.items()))


def exercise_status(exercise: Exercise) -> Status:
    """仅检测实际 raise 占位语句；注释中的 NotImplementedError 不影响判断。

    ready 只表示可以开始测试，不等于正确。语法错误也应进入 --all 以报告失败。
    """
    if not exercise.workspace.is_file():
        return "missing"
    try:
        tree = ast.parse(exercise.workspace.read_text(encoding="utf-8"))
    except SyntaxError:
        return "syntax_error"
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise):
            exception = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            if isinstance(exception, ast.Name) and exception.id == "NotImplementedError":
                return "unfinished"
            if isinstance(exception, ast.Attribute) and exception.attr == "NotImplementedError":
                return "unfinished"
    return "ready"


def redirected_test(exercise: Exercise) -> str:
    """只改被测模块的 import；测试断言、参数化和其他依赖保持原有含义。

    不替换 sys.modules，所以各练习之间不会相互污染，也不会把未完成的基础练习
    注入其他组合模块。当前标准测试统一使用绝对 from ... import ...。
    """
    tree = ast.parse(exercise.standard_test.read_text(encoding="utf-8"))
    replacements = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            if node.module == exercise.source_module:
                node.module = exercise.practice_module
                replacements += 1
    if not replacements:
        raise ValueError(f"{exercise.standard_test} 未找到被测模块导入，已停止以防误测标准答案")
    return ast.unparse(tree) + "\n"


def run_exercises(root: Path, exercises: list[Exercise], pytest_args: list[str]) -> int:
    """在临时目录生成重定向后的标准测试，使用当前解释器启动隔离 pytest 进程。"""
    if not exercises:
        print("没有可测试的练习；完成占位代码后重试，或直接指定一个练习名。", flush=True)
        return 5  # 与 pytest 的“没有收集到测试”状态一致，避免误报通过。
    # 放在项目根目录下，让 pytest 正常发现根 conftest.py 的 seed/CPU fixture。
    with tempfile.TemporaryDirectory(prefix=".practice-tests-", dir=root) as temporary:
        directory = Path(temporary)
        for exercise in exercises:
            destination = directory / f"test_{exercise.name}.py"
            destination.write_text(redirected_test(exercise), encoding="utf-8")
            print(f"[练习] {exercise.workspace.relative_to(root)}", flush=True)
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(root / "pyproject.toml"),
            "--rootdir",
            str(root),
            str(directory),
            "-v",
            *pytest_args,
        ]
        return subprocess.run(command, cwd=root, check=False).returncode


def main(argv: list[str] | None = None, *, root: Path = ROOT) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    pytest_args: list[str] = []
    if "--" in arguments:
        split = arguments.index("--")
        arguments, pytest_args = arguments[:split], arguments[split + 1 :]
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="pytest 参数放在 -- 后，例如：python -m practice.check fundamentals__tensor_ops -- -x -vv",
    )
    parser.add_argument("exercise", nargs="?", help="练习名，或 workspace 中的 .py 文件路径")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--all", action="store_true", help="测试无占位语句的练习；跳过待完成/缺失文件"
    )
    modes.add_argument("--list", action="store_true", help="列出所有练习及占位状态，不执行代码")
    args = parser.parse_args(arguments)
    if args.exercise and (args.all or args.list):
        parser.error("练习名不能与 --all / --list 同时使用")
    if not args.exercise and not (args.all or args.list):
        parser.error("请指定练习名，或使用 --all / --list")
    if args.list and pytest_args:
        parser.error("--list 不接受 pytest 参数")

    root = root.resolve()
    catalog = discover(root)
    if args.list:
        for exercise in catalog.values():
            print(f"{exercise.name:<42} {STATUS_LABELS[exercise_status(exercise)]}")
        return 0
    if args.all:
        selected = [
            exercise
            for exercise in catalog.values()
            if exercise_status(exercise) in ("ready", "syntax_error")
        ]
        print(
            f"选中 {len(selected)} 个练习，跳过 {len(catalog) - len(selected)} 个待完成/缺失文件。",
            flush=True,
        )
    else:
        raw = args.exercise
        name = Path(raw).stem if raw.endswith(".py") else raw
        if name not in catalog:
            parser.error(f"未知练习：{raw}；使用 --list 查看可用名称")
        exercise = catalog[name]
        if ("/" in raw or "\\" in raw) and Path(raw).resolve() != exercise.workspace:
            parser.error(f"路径不匹配，预期：{exercise.workspace}")
        if not exercise.workspace.is_file():
            parser.error(f"练习文件不存在：{exercise.workspace}")
        selected = [exercise]  # 明确指定时不跳过占位，让 pytest 展示尚未实现的位置。
    try:
        return run_exercises(root, selected, pytest_args)
    except (OSError, SyntaxError, ValueError) as error:
        print(f"练习测试启动失败：{error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
