#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cuDNN RNN Forward 推理测试验证脚本
检查 cuDNN 版本和编译兼容性
"""

import os
import subprocess
import sys

def check_cuda_installation():
    """检查 CUDA 安装"""
    print("=== 检查 CUDA 安装 ===")
    
    cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
    print(f"CUDA_HOME: {cuda_home}")
    
    if os.path.exists(cuda_home):
        print(f"✓ CUDA 目录存在: {cuda_home}")
        
        # 检查 nvcc
        nvcc_path = os.path.join(cuda_home, 'bin', 'nvcc')
        if os.path.exists(nvcc_path):
            try:
                result = subprocess.run([nvcc_path, '--version'], 
                                       capture_output=True, text=True)
                if result.returncode == 0:
                    print("✓ nvcc 可用")
                    # 提取版本信息
                    for line in result.stdout.split('\n'):
                        if 'release' in line:
                            print(f"  {line.strip()}")
                            break
                else:
                    print("❌ nvcc 不可用")
            except Exception as e:
                print(f"❌ nvcc 检查失败: {e}")
        else:
            print("❌ nvcc 不存在")
    else:
        print(f"❌ CUDA 目录不存在: {cuda_home}")
    
    return cuda_home

def check_cudnn_installation():
    """检查 cuDNN 安装"""
    print("\n=== 检查 cuDNN 安装 ===")
    
    cudnn_home = os.environ.get('CUDNN_HOME', '/usr/local/cudnn')
    print(f"CUDNN_HOME: {cudnn_home}")
    
    # 检查 cuDNN 头文件
    cudnn_include = os.path.join(cudnn_home, 'include')
    cudnn_header = os.path.join(cudnn_include, 'cudnn.h')
    
    if os.path.exists(cudnn_header):
        print(f"✓ cuDNN 头文件存在: {cudnn_header}")
        
        # 尝试读取版本信息
        try:
            with open(cudnn_header, 'r') as f:
                content = f.read()
                # 查找版本定义
                for line in content.split('\n'):
                    if 'CUDNN_MAJOR' in line and 'define' in line:
                        print(f"  {line.strip()}")
                    elif 'CUDNN_MINOR' in line and 'define' in line:
                        print(f"  {line.strip()}")
                    elif 'CUDNN_PATCHLEVEL' in line and 'define' in line:
                        print(f"  {line.strip()}")
        except Exception as e:
            print(f"❌ 读取头文件失败: {e}")
    else:
        print(f"❌ cuDNN 头文件不存在: {cudnn_header}")
    
    # 检查 cuDNN 库文件
    for lib_path in ['lib/x86_64-linux-gnu', 'lib/x86_64-linux-gnu']:
        cudnn_lib = os.path.join(cudnn_home, lib_path, 'libcudnn.so')
        if os.path.exists(cudnn_lib):
            print(f"✓ cuDNN 库文件存在: {cudnn_lib}")
            break
    else:
        print("❌ cuDNN 库文件不存在")
    
    return cudnn_home

def check_source_code():
    """检查源代码文件"""
    print("\n=== 检查源代码 ===")
    
    source_file = "rnn_performance_test.cpp"
    if os.path.exists(source_file):
        print(f"✓ 源代码文件存在: {source_file}")
        
        # 检查关键函数调用
        with open(source_file, 'r') as f:
            content = f.read()
            
        # 检查是否使用 cudnnRNNForward
        if 'cudnnRNNForward(' in content:
            print("✓ 使用 cudnnRNNForward 接口")
        else:
            print("❌ 未找到 cudnnRNNForward 接口")
            
        # 检查是否使用正确的常量
        if 'CUDNN_RNN_BIAS_ENABLED' in content:
            print("✓ 使用 CUDNN_RNN_BIAS_ENABLED 常量")
        else:
            print("❌ 未找到 CUDNN_RNN_BIAS_ENABLED 常量")
            
        if 'CUDNN_RNN_INPUT_LINEAR_FIRST' in content:
            print("✓ 使用 CUDNN_RNN_INPUT_LINEAR_FIRST 常量")
        else:
            print("❌ 未找到 CUDNN_RNN_INPUT_LINEAR_FIRST 常量")
            
        # 检查工作空间计算
        if 'cudnnGetRNNReserveSpaceSize' in content:
            print("✓ 使用 cudnnGetRNNReserveSpaceSize")
        else:
            print("❌ 未找到 cudnnGetRNNReserveSpaceSize")
            
    else:
        print(f"❌ 源代码文件不存在: {source_file}")

def check_build_system():
    """检查构建系统"""
    print("\n=== 检查构建系统 ===")
    
    # 检查 CMakeLists.txt
    if os.path.exists('CMakeLists.txt'):
        print("✓ CMakeLists.txt 存在")
        
        # 检查 build.sh
        if os.path.exists('build.sh'):
            print("✓ 构建脚本存在")
            if os.access('build.sh', os.X_OK):
                print("✓ 构建脚本可执行")
            else:
                print("❌ 构建脚本不可执行")
        else:
            print("❌ 构建脚本不存在")
    else:
        print("❌ CMakeLists.txt 不存在")

def main():
    """主函数"""
    print("cuDNN RNN Forward 推理测试验证脚本")
    print("=" * 50)
    
    # 检查各个组件
    check_cuda_installation()
    check_cudnn_installation()
    check_source_code()
    check_build_system()
    
    print("\n=== 验证完成 ===")
    print("如果所有检查都通过，可以运行以下命令编译：")
    print("./build.sh")
    print("\n编译成功后运行：")
    print("cd build && ./rnn_performance_test")

if __name__ == "__main__":
    main()