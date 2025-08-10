#!/usr/bin/env python3
"""
LSTM修复验证脚本
测试LSTM权重重排修复后的效果
"""

import subprocess
import os
import sys

def test_lstm_fix():
    """测试LSTM修复"""
    print("=== LSTM修复验证测试 ===")
    
    # 检查是否有可执行文件
    executable_path = "./build_new/rnn_performance_test"
    if not os.path.exists(executable_path):
        print("❌ 可执行文件不存在，请先编译项目")
        return False
    
    # 运行测试，只测试LSTM配置
    try:
        result = subprocess.run([
            executable_path,
            "--operator_type", "LSTM",
            "--max_tests", "10"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ LSTM测试运行成功")
            
            # 分析输出
            output = result.stdout
            if "NaN" in output:
                print("❌ 仍然存在NaN值")
                return False
            elif "FAILED" in output:
                print("❌ 存在失败的测试")
                return False
            else:
                print("✅ LSTM测试通过，没有发现NaN值")
                return True
        else:
            print(f"❌ 测试失败，返回码: {result.returncode}")
            print(f"错误输出: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 测试超时")
        return False
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False

if __name__ == "__main__":
    success = test_lstm_fix()
    sys.exit(0 if success else 1)