#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script to verify the initial hidden state fix
Specifically tests the scenario that was failing in TEST#3
"""

import numpy as np
import subprocess
import sys
import os

def run_specific_test():
    """Run the specific test that was failing"""
    print("=== Testing Initial Hidden State Fix ===")
    print("Running TEST#3: RNN forward Tanh with initial_h")
    print()
    
    # Change to build directory
    os.chdir('/home/yiyu/cuda/cudnn/cudnn_test/build')
    
    # Create a minimal test script that only runs TEST#3
    test_script = """
#include <iostream>
#include <vector>
#include <chrono>
#include "rnn_performance_test.h"

int main() {
    std::cout << "=== Testing Initial Hidden State Fix ===" << std::endl;
    
    // Create test configuration for TEST#3
    TestConfig config;
    config.operator_type = "RNN";
    config.direction = "forward";
    config.activation = "Tanh";
    config.seq_length = 4;
    config.batch_size = 2;
    config.input_size = 8;
    config.hidden_size = 16;
    config.include_sequence_lens = false;
    config.include_initial_h = true;
    config.include_initial_c = false;
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Operator: " << config.operator_type << std::endl;
    std::cout << "  Direction: " << config.direction << std::endl;
    std::cout << "  Activation: " << config.activation << std::endl;
    std::cout << "  Shape: seq=" << config.seq_length << ", batch=" << config.batch_size 
              << ", input=" << config.input_size << ", hidden=" << config.hidden_size << std::endl;
    std::cout << "  Optional inputs: initial_h=" << config.include_initial_h << std::endl;
    std::cout << std::endl;
    
    // Run the test
    RNNPerformanceTest test;
    TestResult result = test.RunTest(config, 3);
    
    std::cout << "Result:" << std::endl;
    std::cout << "  Cosine Similarity: " << result.cosine_similarity << std::endl;
    std::cout << "  Status: " << (result.passed ? "PASSED" : "FAILED") << std::endl;
    
    if (result.passed) {
        std::cout << std::endl;
        std::cout << "✅ SUCCESS: Initial hidden state fix is working!" << std::endl;
        std::cout << "✅ TEST#3 precision issue has been resolved." << std::endl;
        return 0;
    } else {
        std::cout << std::endl;
        std::cout << "❌ FAILED: Initial hidden state fix is not working." << std::endl;
        return 1;
    }
}
"""
    
    # Write the test script
    with open('test_initial_h_fix.cpp', 'w') as f:
        f.write(test_script)
    
    # Compile the test
    compile_cmd = ['g++', '-std=c++17', '-I../', '-I/usr/local/cuda/include', 
                   '-I/usr/local/onnxruntime-gpu/include', '-L/usr/local/cuda/lib64',
                   '-L/usr/local/onnxruntime-gpu/lib', '-lcudart', '-lcudnn', 
                   '-lonnxruntime', '-o', 'test_initial_h_fix', 'test_initial_h_fix.cpp', 
                   'rnn_performance_test.cpp']
    
    try:
        result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print("❌ Compilation failed:")
            print(result.stderr)
            return False
        
        # Run the test
        test_result = subprocess.run(['./test_initial_h_fix'], capture_output=True, text=True, timeout=30)
        print(test_result.stdout)
        if test_result.stderr:
            print("STDERR:", test_result.stderr)
        
        return test_result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ Test timed out")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main function"""
    print("Initial Hidden State Fix Verification")
    print("=" * 50)
    
    success = run_specific_test()
    
    if success:
        print("\n🎉 VERIFICATION COMPLETE")
        print("The initial hidden state fix is working correctly!")
        print("TEST#3 precision issue has been resolved.")
    else:
        print("\n❌ VERIFICATION FAILED")
        print("The initial hidden state fix needs further investigation.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())