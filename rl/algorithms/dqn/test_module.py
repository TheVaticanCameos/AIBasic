import torch

from rl.algorithms.dqn.implementation import DQN, QNetwork
from rl.buffers.replay.implementation import TransitionBatch


def test_double_dqn_selection_terminal_and_target_gradients() -> None:
    agent = DQN(2, 2, gamma=0.5)
    with torch.no_grad():
        for p in agent.online.parameters():
            p.zero_()
        for p in agent.target.parameters():
            p.zero_()
        agent.online.advantage.bias.copy_(torch.tensor([2.0, 1.0]))
        agent.target.advantage.bias.copy_(torch.tensor([3.0, 9.0]))
    batch = TransitionBatch(
        torch.zeros(2, 2),
        torch.tensor([0, 1]),
        torch.ones(2),
        torch.zeros(2, 2),
        torch.tensor([False, True]),
        torch.tensor([True, False]),
    )
    torch.testing.assert_close(agent.targets(batch), torch.tensor([2.5, 1.0]))
    agent.double = False
    torch.testing.assert_close(agent.targets(batch), torch.tensor([5.5, 1.0]))
    agent.update(batch)
    assert all(p.grad is None for p in agent.target.parameters())


def test_dueling_value_is_mean_q_and_greedy_actions() -> None:
    network = QNetwork(3, 4, dueling=True)
    obs = torch.randn(2, 3)
    assert network.value is not None
    torch.testing.assert_close(
        network(obs).mean(-1), network.value(network.features(obs)).squeeze(-1)
    )
    agent = DQN(3, 4)
    torch.testing.assert_close(agent.act(obs), agent.online(obs).argmax(-1))
