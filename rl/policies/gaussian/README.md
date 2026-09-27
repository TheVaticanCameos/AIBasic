# Gaussian Policy 与 Tanh 修正

优先级：**P1**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

u=μ+σϵ，a=tanh(u)；log π(a)=Σ(log N(u)-log(1-tanh²u))。

## Tensor 约定

obs:[B,O] → mean/log_std/actions:[B,A]；log_prob 对动作维求和后 [B]。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

网络输出均值和截断 log_std；rsample 重参数采样；使用稳定 Jacobian 修正；可选不 squash。

## 易错点与支持边界

log_prob 不能对动作维取 mean。输出密度定义在标准 [-1,1] 空间；若计算环境尺度密度还要减 log scale 的和。

## 复杂度

MLP 加 O(BA) 采样/变换；网络参数随隐藏层维度增长。

## 接口与依赖

`forward(self, observations: Tensor) -> tuple[Tensor, Tensor]`

observations [B,O] -> (mean, log_std)，各 [B,A]。

`sample(self, observations: Tensor, squash: bool=True, deterministic: bool=False) -> tuple[Tensor, Tensor]`

obs [B,O] -> actions [B,A], log_probs [B]；squash 时动作范围 [-1,1]。

log_prob 对动作维求和。环境边界缩放交给外层；本模块密度定义在标准动作空间。

仓库内依赖：无；可单独复制 implementation.py，仅依赖标准库/PyTorch。

## 运行与验证

在仓库根目录运行：

```bash
python -m rl.policies.gaussian.implementation
python -m pytest rl/policies/gaussian/test_module.py
```

测试关注：

- `test_tanh_density_matches_transformed_distribution_and_rsample_gradient`
- `test_extreme_mean_has_finite_corrected_log_prob`

## 面试追问与练习

sample 与 rsample 有何不同？tanh 为什么改变概率密度？

[独立手写题与起始接口](../../../practice/prompts/rl__policies__gaussian.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
