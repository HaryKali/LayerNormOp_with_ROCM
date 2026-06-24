# Easy LayerNorm (ROCm HIP 实现)

本目录实现了一个基于 ROCm HIP 的自定义 LayerNorm 前向算子， 正在逐步完善当中

## 1. 模块简介

这次实现通过 PyTorch C++ Extension 机制，将 HIP Kernel 注册为 Python 可调用的模块。核心功能是对二维张量 `[rows, cols]` 的每一行进行 Layer Normalization（标准化）。

当前版本主要特点是仅实现前向传播，使用共享内存和树形归约来实现高效的行内归约计算，Block 大小固定为 256，并通过官方 LayerNorm 进行正确性验证。 

## 

## 2. 算法介绍

这个算子对输入矩阵的每一行分别做标准化处理。假设输入是形状为 `[rows, cols]` 的二维张量，对于每一行 $x$，我们先计算这一行的平均值 $\text{mean}$ 和方差 $\text{var}$，然后把这行数据变成均值为 0、方差接近 1 的结果。

核心计算公式如下：

$$
\text{mean} = \frac{1}{N} \sum_{i=1}^{N} x_i
$$

$$
\text{var} = \frac{1}{N} \sum_{i=1}^{N} x_i^2 - \text{mean}^2
$$

$$
y_i = \frac{x_i - \text{mean}}{\sqrt{\text{var} + \epsilon}}
$$

其中 $N$ 表示每行的元素个数（即 `cols`），$\epsilon$ 是一个很小的正数，防止方差为 0 时出现除以零的情况。

  在 GPU 上实现时，每一个 Block 负责处理一行数据。线程先用 Grid-Stride Loop 把这一行的数据累加起来，得到 `sum` 和 `sum_sq`，然后通过树形归约快速求出整行的总和，再算出 $\text{mean}$ 和 $\text{var}$，最后再把结果写回输出显存。

---

## 3. 与官方 `torch.nn.LayerNorm` 的对比

这次实现来的LayerNrom实际上是官方的简化版，整体的对比如下，之后会逐步改进

| 算子                  | 官方 `torch.nn.LayerNorm` | 本自定义实现 (`my_layernorm_backend`) |
| ------------------- | ----------------------- | ------------------------------- |
| **支持反向传播**          | 支持                      | 不支持                             |
| **gamma / beta 参数** | 支持（默认开启）                | 不支持                             |
| **输入维度支持**          | 支持任意维度                  | 支持 2D `[rows, cols]`            |
| **Block 大小**        | 动态优化                    | 固定 256                          |

---

## 4. 正确性验证方法

我们用 PyTorch 官方的 LayerNorm 作为标准答案来检查自己的实现的正确性。测试脚本会把官方算出来的结果和我们自己算的结果分别打印出来，然后算出两者之间的最大误差和平均误差。只要最大误差小于 0.0001，就认为通过测试。测试脚本会跑好几组不同大小的数据来验证。

---

## 5. 技术实现要点

这个实现主要靠共享内存和树形归约来加快计算。共享内存让同一个 Block 里的线程能快速互相传数据，树形归约用多轮配对的方式快速把一整行的数据加起来。Kernel 启动时会分配共享内存空间，最后通过 PyTorch 的扩展机制把 HIP 代码变成 Python 可以调用的模块。具体来说i可以分为以下4步

- **1. 局部累加（Grid-Stride Loop）**：
  每个线程通过网格步长循环遍历当前行的所有列，在寄存器中同步累加自己负责的元素之和（`local_sum`）与平方之和（`local_sum_sq`）。该设计允许线程块处理超出自身线程数量的任意列宽。

- **2. 共享内存配置**：
  各线程将计算出的局部累加值分别写入动态共享内存的对应区间（前半段存元素和，后半段存平方和）。随后执行 `__syncthreads()` 阻挡，确保 Block 内所有线程数据完全写入。

- **3. 树形归约（Tree Reduction）**：
  在 Block 内进行多轮步长（Stride）减半的折叠累加。每轮归约完成后均进行一次线程同步，以消除数据竞争。最终在共享内存的特定位置（索引 `0` 和 `blockDim.x`）递推出整行的全局总和与平方总和。

- **4. 参数计算与结果写回**：
  根据归约得到的整行总和计算出该行的均值与方差。最后，各线程再次循环读取输入显存，套用 LayerNorm 标准化公式计算最终结果，并直接写回全局显存。



---

## 6. 使用算子

### 编译

```bash
cd custom_layernorm
python setup.py build_ext --inplace
```

### Python 调用示例

```python
import torch
import my_layernorm_backend

x = torch.randn(128, 1024, device="cuda", dtype=torch.float32)
y = my_layernorm_backend.layernorm_forward(x, eps=1e-5)
```

### 正确性测试

```bash
python test_layernorm.py
```

测试脚本会分别打印官方实现和自定义实现的输出，并计算最大/平均绝对误差。

---

## 7. 当前限制

当前实现仅支持前向传播，无法用于训练，同时不支持 gamma 和 beta 可学习参数，仅支持二维输入，Block 大小固定为 256，且方差使用有偏估计。

---

## 8. 后续可扩展方向

后续可扩展的方向包括增加 gamma 和 beta 支持、实现反向传播 Kernel 以支持训练、使用 torch.autograd.Function 封装完整算子、支持多维度输入以及进行性能 benchmark 对比。

---

## 9. 相关文件

| 文件                      | 说明                        |
| ----------------------- | ------------------------- |
| `layernorm_kernel.hip`  | 核心 HIP Kernel 实现          |
| `layernorm_wrapper.cpp` | PyTorch C++ Extension 包装层 |
| `setup.py`              | 编译脚本                      |
| `test_layernorm.py`     | 正确性测试脚本（使用官方实现作为基准）       |

---

## 10. 结果预览：
<img width="1582" height="1126" alt="image" src="https://github.com/user-attachments/assets/f2885daf-27e3-451e-9bfe-a46ac588da54" />
