# cuDNN RNN 性能测试套件

这是一个针对 cuDNN RNN 前向接口 (`cudnnRNNForward`) 的全面性能和功能测试套件。测试涵盖了 RNN、GRU 和 LSTM 算法的各种配置和形状。

## 功能特性

- **全面测试**: 测试 RNN、GRU 和 LSTM 算法
- **多种形状**: 覆盖小、中、大三种输入形状
- **性能测量**: 精确的计时和 GFLOPS 计算
- **500+ 测试用例**: 广泛的微基准测试覆盖
- **错误处理**: 健壮的错误检查和报告
- **输出验证**: 验证输出的 NaN/Inf 值

## 测试覆盖范围

### 算法
- ReLU 激活的 RNN
- Tanh 激活的 RNN
- GRU (门控循环单元)
- LSTM (长短期记忆网络)

### 形状
- **小型**: Batch 大小 1-8，序列长度 10-100，输入大小 32-256
- **中型**: Batch 大小 16-64，序列长度 128-512，输入大小 512-1024
- **大型**: Batch 大小 128-256，序列长度 1024-2048，输入大小 2048-4096

### 配置
- 单向和双向 RNN
- 1-8 层
- 不同的输入/隐藏层大小比例
- 边界情况和性能导向的配置

## 构建要求

- NVIDIA CUDA 工具包 (11.0 或更高版本)
- cuDNN 库 (8.0 或更高版本)
- 支持 C++17 的 GCC 编译器
- Linux 操作系统
- CMake 3.18 或更高版本

## 构建方法

```bash
# 创建构建目录
mkdir build && cd build

# 检查依赖项
cmake .. -DCMAKE_BUILD_TYPE=Release

# 构建测试程序
cmake --build .

# 构建调试版本
cmake .. -DCMAKE_BUILD_TYPE=Debug && cmake --build .

# 构建性能分析版本
cmake .. -DCMAKE_BUILD_TYPE=Profile && cmake --build .

# 运行测试
ctest
```

## 运行测试

```bash
# 运行测试
cd build
./rnn_performance_test

# 或使用 CTest
ctest --verbose
```

## 环境变量

您可以使用以下环境变量自定义构建：

```bash
# 设置 CUDA 安装路径
export CUDA_HOME=/usr/local/cuda

# 设置 cuDNN 安装路径
export CUDNN_HOME=/usr/local/cudnn
```

或使用 CMake 缓存变量：

```bash
# 使用自定义路径配置
cmake .. -DCUDA_HOME=/path/to/cuda -DCUDNN_HOME=/path/to/cudnn
```

## 输出

测试产生详细的输出，包括：

1. **系统信息**
   - cuDNN 版本
   - CUDA 运行时版本
   - 设备属性

2. **测试执行**
   - 单个测试结果
   - 执行时间（毫秒）
   - 性能（GFLOPS）
   - 输出验证状态

3. **摘要统计**
   - 总测试数
   - 通过/失败计数
   - 成功率
   - 平均性能
   - 按算法分类的性能分析

## 示例输出

```
cuDNN RNN 性能和功能测试
=============================================
cuDNN 版本: 8900
CUDA 运行时版本: 12000
设备: NVIDIA GeForce RTX 4090
计算能力: 8.9

生成了 540 个测试配置

测试 1/540: RNN_RELU BS=1 SL=10 IS=32 HS=32 NL=1 BI=否
  时间: 0.123 毫秒
  性能: 0.85 GFLOPS
  输出验证: 通过

测试 2/540: RNN_RELU BS=1 SL=10 IS=32 HS=32 NL=2 BI=否
  时间: 0.145 毫秒
  性能: 1.02 GFLOPS
  输出验证: 通过

...

测试摘要
============
总测试数: 540
通过: 540
失败: 0
成功率: 100.00%
平均性能: 125.67 GFLOPS

按 RNN 模式分类的性能
=======================
RNN_RELU: 平均=145.23 GFLOPS, 最小=0.85 GFLOPS, 最大=892.34 GFLOPS
RNN_TANH: 平均=138.91 GFLOPS, 最小=0.92 GFLOPS, 最大=856.78 GFLOPS
GRU: 平均=112.45 GFLOPS, 最小=0.78 GFLOPS, 最大=723.56 GFLOPS
LSTM: 平均=98.76 GFLOPS, 最小=0.65 GFLOPS, 最大=645.23 GFLOPS

测试成功完成！
```

## 测试详情

### 性能测量
- 使用高分辨率计时器进行精确测量
- 包含预热迭代以考虑 GPU 初始化
- 报告多次迭代的平均时间
- 基于理论操作次数计算 GFLOPS

### 功能测试
- 验证输出张量的 NaN/Inf 值
- 测试不同的 RNN 配置
- 包含边界情况和压力测试

### 内存管理
- 正确的 CUDA 内存分配和释放
- 工作空间和保留空间管理
- 内存不足情况的错误处理

## 错误处理

测试包含全面的错误处理：
- CUDA 运行时错误
- cuDNN API 错误
- 内存分配失败
- 无效配置检测

## 自定义

您可以在 `generate_test_configs()` 函数中修改测试配置：
- 添加更多测试用例
- 调整形状范围
- 更改算法参数
- 修改迭代次数

## 故障排除

### 常见问题

1. **CUDA/cuDNN 未找到**
   - 确保 CUDA_HOME 和 CUDNN_HOME 设置正确
   - 验证 CMake 配置中的库路径
   - 检查 CMake 输出中的依赖信息

2. **内存不足**
   - 减小批量大小或序列长度
   - 在内存更大的 GPU 上测试

3. **编译错误**
   - 检查 GCC 版本（需要 C++17 支持）
   - 验证 CUDA/cuDNN 安装
   - 确保 CMake 版本 >= 3.18

### 调试模式

使用调试符号构建以调试问题：

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug && cmake --build .
gdb ./build/rnn_performance_test
```

### CMake 配置

其他 CMake 选项：

```bash
# 设置构建类型
cmake .. -DCMAKE_BUILD_TYPE=Release

# 显式设置 cuDNN 路径
cmake .. -DCUDNN_HOME=/path/to/cudnn

# 清理构建
rm -rf build && mkdir build && cmake .. && cmake --build .
```

## 性能说明

- 性能根据 GPU 架构差异很大
- 较大的批量大小通常产生更好的性能
- 双向 RNN 比单向慢约 2 倍
- LSTM 由于更复杂的操作通常最慢
- 性能以 GFLOPS（每秒十亿次浮点运算）衡量

## 许可证

本测试套件按原样提供，用于性能评估目的。

## 技术支持

如有问题或建议，请参考：
- NVIDIA CUDA 官方文档
- cuDNN 开发者指南
- CMake 官方文档

---

**注意**: 此测试套件需要兼容的 NVIDIA GPU 和正确的 CUDA/cuDNN 安装才能正常运行。