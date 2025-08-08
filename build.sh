#!/bin/bash

# cuDNN RNN Forward 推理性能测试编译脚本
# 适用于 cuDNN 8.9.7 版本

echo "=== cuDNN RNN Forward 推理性能测试编译脚本 ==="
echo "cuDNN 版本: 8.9.7"
echo "构建时间: $(date)"
echo

# 检查必要的环境变量
if [ -z "$CUDA_HOME" ]; then
    echo "警告: CUDA_HOME 未设置，使用默认路径 /usr/local/cuda"
    CUDA_HOME="/usr/local/cuda"
fi

if [ -z "$CUDNN_HOME" ]; then
    echo "警告: CUDNN_HOME 未设置，使用默认路径 /usr/local/cudnn"
    CUDNN_HOME="/usr/local/cudnn"
fi

# 检查文件是否存在
echo "检查必要文件..."
if [ ! -f "rnn_performance_test.cpp" ]; then
    echo "错误: rnn_performance_test.cpp 不存在"
    exit 1
fi

if [ ! -f "CMakeLists.txt" ]; then
    echo "错误: CMakeLists.txt 不存在"
    exit 1
fi

echo "✓ rnn_performance_test.cpp 存在"
echo "✓ CMakeLists.txt 存在"

# 创建构建目录
echo
echo "创建构建目录..."
if [ ! -d "build" ]; then
    mkdir -p build
    echo "✓ 构建目录已创建"
else
    echo "✓ 构建目录已存在"
fi

# 进入构建目录
cd build

# 运行 CMake 配置
echo
echo "运行 CMake 配置..."
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDNN_HOME: $CUDNN_HOME"
export ONNXRUNTIME_HOME=/usr/local/onnxruntime-gpu
if cmake .. -DCMAKE_BUILD_TYPE=Debug \
            -DCUDA_HOME="$CUDA_HOME" \
            -DCUDNN_HOME="$CUDNN_HOME" \
            -DONNXRUNTIME_HOME="${ONNXRUNTIME_HOME}" \
            -DCMAKE_CUDA_ARCHITECTURES=80 2>&1; then
    echo "✓ CMake 配置成功"
else
    echo "❌ CMake 配置失败"
    echo "请检查 CUDA 和 cuDNN 安装路径"
    exit 1
fi

# 编译项目
echo
echo "开始编译..."
rm -rf *.onnx
rm -rf CMakeCache.txt CMakeFiles  *.cmake
rm -rf Makefile
#make clean
cmake ..
if make -j$(nproc) 2>&1; then
    echo "✓ 编译成功"
    
    # 检查生成的可执行文件
    if [ -f "rnn_performance_test" ]; then
        echo "✓ 可执行文件已生成: rnn_performance_test"
        echo
        echo "构建信息:"
        ls -la rnn_performance_test
        echo
        echo "运行命令:"
        echo "./rnn_performance_test"
    else
        echo "❌ 可执行文件未生成"
        exit 1
    fi
else
    echo "❌ 编译失败"
    echo "请检查编译错误信息"
    exit 1
fi

echo
echo "=== 构建完成 ==="