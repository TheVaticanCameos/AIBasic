# RNN / LSTM / GRU Cell

优先级：**P1**。先理解公式、形状和梯度，再独立重写实现。

## 核心公式与用途

RNN: h=tanh(Wx+Uh)；LSTM: c=f*c_prev+i*g,h=o*tanh(c)；GRU: h=(1-z)*n+z*h_prev。

## Tensor 约定

x:[B,I]，hidden/cell:[B,H]；LSTM 门 [B,4H]，GRU 门 [B,3H]。

全库约定：B=batch，T=时间/序列，D=特征维，H=head 或图像高度（以局部注释为准），A=动作数/动作维，O=观测维或输出维（以接口为准）。浮点默认 float32；索引默认 int64，mask/flags 为 bool。未特别说明时，参与运算的张量应有匹配的 device/dtype。

## 实现步骤

分别计算输入与 hidden 投影；按 gate 切块；应用激活后更新状态。三个 Cell 均可独立使用。

## 易错点与支持边界

LSTM 顺序 i,f,g,o；GRU 顺序 r,z,n，reset 乘在 hidden 投影后，与 PyTorch 对齐。只实现单步 cell。

## 复杂度

每步 O(B*(IH+H²))，LSTM/GRU 有额外常数门数。

## 接口与依赖

`forward(self, x: Tensor, h: Tensor) -> Tensor`

x [B,I], h [B,H] -> h_new [B,H]。

`forward(self, x: Tensor, state: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]`

x [B,I], state=(h,c)，各 [B,H] -> (h_new,c_new)，各 [B,H]。

`forward(self, x: Tensor, h: Tensor) -> Tensor`

x [B,I], h [B,H] -> h_new [B,H]；reset gate 乘在 hidden 投影之后。

仓库内依赖：无；可单独复制 implementation.py，仅依赖标准库/PyTorch。

## 运行与验证

在仓库根目录运行：

```bash
python -m layers.recurrent.implementation
python -m pytest layers/recurrent/test_module.py
```

测试关注：

- `test_cells_match_pytorch_outputs_and_gradients`

## 面试追问与练习

LSTM cell state 与 hidden state 有何区别？GRU reset gate 放在矩阵乘法前后是否等价？

[独立手写题与起始接口](../../practice/prompts/layers__recurrent.md)。掌握程度填写到根目录 ROADMAP.md；标准答案已实现不代表自己已掌握。
