# Debug 题

每题先解释错误，再写最小反例，最后修复并验证。建议每题 5–10 分钟。

## 1. Value loss 隐式广播

```python
prediction = value_net(obs)  # [B,1]
target = returns  # [B]
loss = ((prediction - target) ** 2).mean()
```

要求：给出 B=2 的数字反例；修复时保持 B=1 也正确。参考 [回归损失](../losses/regression/README.md)。

## 2. Actor 丢失动作梯度

```python
action = actor(obs)  # [B,A]
with torch.no_grad():
    score = critic(obs, action)  # [B]
loss = -score.mean()
loss.backward()
```

要求：critic 参数不获得新梯度，但 actor 必须能更新；说明为什么冻结与 no_grad 不等价。参考 [参数冻结](../rl/updates/target/README.md)。

## 3. PPO 只对正优势想当然地裁剪

```python
ratio = (new_log_prob - old_log_prob).exp()  # [B]
loss = -(ratio.clamp(0.8, 1.2) * advantage).mean()
```

要求：分别构造正/负 advantage、ratio 高于/低于边界的四个例子，说明哪种策略变化应该继续受到惩罚。参考 [PPO objective](../rl/objectives/ppo/README.md)。

## 4. Time limit 被当作真正终止

```python
done = terminated | truncated  # bool [B]
target = reward + gamma * (~done) * next_value
```

要求：解释 CartPole/Pendulum 时间限制与 MDP terminal 的区别；给出 bootstrap mask 与 GAE trace mask。参考 [RL 数据流](../notes/rl_dataflow.md)。

## 5. KV Cache 新 query 看不到历史

```python
mask = torch.ones(new_length, total_length, dtype=torch.bool).tril()
```

要求：历史长度为 3，新块长度为 2，画出正确的 2×5 mask；比较 full sequence 与分块输出。参考 [KV Cache](../attention/kv_cache/README.md)。

## 6. AdamW 写成 Adam + L2

```python
gradient = parameter.grad + weight_decay * parameter
first = beta1 * first + (1 - beta1) * gradient
```

要求：说明 weight decay 如何进入了自适应分母；写出真正解耦衰减的位置，并与官方 AdamW 连续对照 5 次更新。参考 [Adam/AdamW](../optimizers/adam/README.md)。

