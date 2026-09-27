"""固定随机种子，小张量 CPU 测试使用单线程。"""

import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic_tests() -> None:
    torch.manual_seed(7)
    torch.set_num_threads(1)
