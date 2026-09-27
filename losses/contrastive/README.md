# 双向 InfoNCE

优先级：**P1**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

logits_ij=cos(x_i,y_j)/τ；对行和列各计算一次 CE 后平均。

## Tensor 约定

x/y:[B,D] → logits:[B,B]；同索引为正对；B>=2。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

L2 归一化、计算两组 embedding 相似度、构造 arange 标签、双向 CE。

## 易错点与支持边界

temperature>0；batch 内其他样本均视为负样本，不处理重复语义正样本或跨设备聚合。

## 复杂度

O(B²D) 计算，O(B²) score 内存。

## 接口与依赖

`info_nce(x: Tensor, y: Tensor, temperature: float=0.1) -> Tensor`

x/y [B,D] -> scalar []；要求 B>=2，temperature>0。

仓库内依赖：[losses.classification.implementation](../classification/implementation.py)

## 运行与验证

在仓库根目录运行：

```bash
python -m losses.contrastive.implementation
python -m pytest losses/contrastive/test_module.py
```

测试关注：

- `test_matched_pairs_better_and_symmetric`

## 面试追问与练习

温度影响怎样的梯度？为什么双向 loss 更适合对称匹配？

[独立手写题与起始接口](../../practice/prompts/losses__contrastive.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
