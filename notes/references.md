# 官方参考资料

以下资料用于核对 API 语义与学习原算法。仓库采用自己展开的教学实现，接口支持范围见各单元 README。

- [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)：对照 Attention 数值与 mask；SDPA bool mask 的 True 表示允许参与。
- [PyTorch DQN 教程](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning)：CartPole 采样、replay 与目标网络更新的参考。
- [PyTorch BatchNorm 源码与文档](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/batchnorm.py)：训练前向方差与 running variance 的估计约定。
- [Gymnasium Time Limits](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)：区分 termination 与 truncation，后者通常仍需 bootstrap。
- [Gymnasium Env API](https://gymnasium.farama.org/api/env/)：reset/step 返回值与环境生命周期。
- [Spinning Up：策略优化](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)：policy gradient 目标与 baseline。
- [Spinning Up：SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html)：双 Q、熵正则目标与 squashed Gaussian 策略。
- [Spinning Up SAC PyTorch core](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py)：稳定 tanh log-Jacobian 的实现参考。

