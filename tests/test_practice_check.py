"""验证测试入口确实执行学生代码、保留 fixture/参数化，并传播失败状态。"""

import ast
from pathlib import Path

import pytest

from practice.check import discover, exercise_status, main, redirected_test, run_exercises


@pytest.fixture
def practice_repo(tmp_path: Path) -> Path:
    files = {
        "pyproject.toml": '[tool.pytest.ini_options]\npythonpath = ["."]\naddopts = "--import-mode=importlib -q"\n',
        "conftest.py": "import pytest\n@pytest.fixture\ndef offset():\n    return 7\n",
        "helper.py": "VALUE = 3\n",
        "fundamentals/demo/implementation.py": 'raise RuntimeError("不得测试标准答案")\n',
        "fundamentals/demo/test_module.py": (
            "import pytest\n"
            "from fundamentals.demo.implementation import answer\n"
            "from helper import VALUE\n"
            '@pytest.mark.parametrize("x", [1, 2])\n'
            "def test_answer(x, offset):\n"
            "    assert answer(x) == x + VALUE + offset\n"
        ),
        "practice/__init__.py": "",
        "practice/workspace/__init__.py": "",
        "practice/workspace/fundamentals__demo.py": "def answer(x):\n    return x + 10\n",
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return tmp_path


def test_rewrites_only_target_import(practice_repo: Path) -> None:
    exercise = discover(practice_repo)["fundamentals__demo"]
    tree = ast.parse(redirected_test(exercise))
    modules = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert modules == ["practice.workspace.fundamentals__demo", "helper"]
    exercise.standard_test.write_text("def test_nothing():\n    assert True\n")
    with pytest.raises(ValueError, match="未找到被测模块导入"):
        redirected_test(exercise)


@pytest.mark.parametrize(
    "source,status",
    [
        ("# raise NotImplementedError\ndef answer(x): return x\n", "ready"),
        ('def answer(x): raise NotImplementedError("练习")\n', "unfinished"),
        ("def answer(x): raise NotImplementedError\n", "unfinished"),
        ("def answer(\n", "syntax_error"),
    ],
)
def test_status_uses_syntax_not_comment_text(practice_repo: Path, source: str, status: str) -> None:
    exercise = discover(practice_repo)["fundamentals__demo"]
    exercise.workspace.write_text(source, encoding="utf-8")
    assert exercise_status(exercise) == status
    exercise.workspace.unlink()
    assert exercise_status(exercise) == "missing"


@pytest.mark.parametrize(
    "source,expected_code",
    [
        ("def answer(x): return x + 10\n", 0),
        ("def answer(x): return -100\n", 1),
        ("def another_name(x): return x + 10\n", 2),
    ],
)
def test_isolated_pytest_runs_student_code_and_cleans_up(
    practice_repo: Path, source: str, expected_code: int, capfd: pytest.CaptureFixture[str]
) -> None:
    exercise = discover(practice_repo)["fundamentals__demo"]
    exercise.workspace.write_text(source, encoding="utf-8")
    original_test = exercise.standard_test.read_text()
    result = run_exercises(practice_repo, [exercise], [])
    assert result == expected_code
    output = capfd.readouterr().out
    if expected_code == 0:
        assert "2 passed" in output  # 参数化与根目录 fixture 均实际生效。
    assert exercise.standard_test.read_text() == original_test
    assert exercise.workspace.read_text() == source
    assert not list(practice_repo.glob(".practice-tests-*"))


def test_list_and_all_skip_placeholders_without_false_success(
    practice_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exercise = discover(practice_repo)["fundamentals__demo"]
    exercise.workspace.write_text("def answer(x): raise NotImplementedError\n")
    assert main(["--list"], root=practice_repo) == 0
    assert "待完成" in capsys.readouterr().out
    assert main(["--all"], root=practice_repo) == 5
    assert "没有可测试的练习" in capsys.readouterr().out


def test_all_and_file_path_entrypoints(practice_repo: Path) -> None:
    assert main(["--all", "--", "-k", "answer"], root=practice_repo) == 0
    exercise = discover(practice_repo)["fundamentals__demo"]
    assert main([str(exercise.workspace)], root=practice_repo) == 0


def test_unknown_exercise_is_usage_error(practice_repo: Path) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["does_not_exist"], root=practice_repo)
    assert caught.value.code == 2
