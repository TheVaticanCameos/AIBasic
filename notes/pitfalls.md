# 易错点与排错

| 现象 | 优先检查 |
|---|---|
| loss 形状看似正常但训练异常 | `[B,1]` 与 `[B]` 是否广播成 `[B,B]` |
| B=1 时出错 | 是否使用无参数 `squeeze()` |
| attention 输出 NaN | 是否存在全遮挡行、mask 是否反了、logits 是否溢出 |
| Transformer 偷看未来 | causal mask 是否沿正确轴、缓存 query offset 是否正确 |
| value/actor 梯度互相污染 | advantage、target、old_log_prob 是否 detach |
| actor 完全没有梯度 | 是否给 critic 的整个前向加了 no_grad |
| replay 内容随环境变化 | add 时是否复制数据，是否保存了环境复用的数组引用 |
| PPO 第二个 epoch 报计算图已释放 | rollout 是否保存了旧策略计算图 |
| SAC loss 不稳定 | tanh Jacobian、动作维求和、温度符号与梯度边界 |
| 梯度累积与大 batch 不等价 | microbatch 权重、BN/Dropout、跨样本损失 |
| AdamW 与官方参数差异逐步增大 | decay 是否解耦、step 是否按参数计数、eps 在何处 |
| BN eval 结果不对 | running buffers 是否更新、训练/推理方差约定是否一致 |

建议按顺序排查：

1. 用 B=1 与 B>1 的小数据打印 shape、dtype、device。
2. 检查 finite、归约轴、mask 含义与边界。
3. 选一组能手算的数据验证值。
4. 检查哪些参数应有/不应有梯度；必要时用 float64 gradcheck。
5. 与官方实现对照时统一权重、训练模式、随机性与选项。
6. 最后才看训练曲线，区分计算错误与采样/超参数问题。

PyTorch 的 `eval()` 不关闭 autograd；`no_grad()` 也不自动关闭 Dropout 或切换 BatchNorm running statistics。评估通常两者都需要。

