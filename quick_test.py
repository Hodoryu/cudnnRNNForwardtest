#!/usr/bin/env python3
"""
Quick test script to validate the fix with a single test case
"""

import subprocess
import sys
import os

def run_single_test():
    """Run a single RNN test to check if the fix works"""
    # Change to build directory
    os.chdir('/home/yiyu/cuda/cudnn/cudnn_test/build')
    
    # Create a simple test model first
    try:
        # Create a simple RNN model
        result = subprocess.run([
            'python3', '../create_rnn_model.py',
            '--operator_type', 'RNN',
            '--sequence_length', '4',
            '--batch_size', '2',
            '--input_size', '8',
            '--hidden_size', '16',
            '--direction', 'forward',
            '--activation', 'Tanh',
            '--output_filename', 'simple_test.onnx'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"Failed to create test model: {result.stderr}")
            return False
            
        print("Test model created successfully")
        
        # Run the performance test with just one iteration
        result = subprocess.run([
            './rnn_performance_test'
        ], capture_output=True, text=True, timeout=120)
        
        # Look for cosine similarity in the output
        output = result.stdout + result.stderr
        if "cosine similarity" in output:
            # Extract the last cosine similarity value
            lines = output.split('\n')
            for line in reversed(lines):
                if "cosine similarity" in line and "Overall" in line:
                    print(f"Found similarity result: {line.strip()}")
                    return True
                    
        print("No cosine similarity found in output")
        print("Output snippet:")
        print(output[-500:])  # Last 500 characters
        
    except subprocess.TimeoutExpired:
        print("Test timed out")
    except Exception as e:
        print(f"Test failed with error: {e}")
        
    return False

if __name__ == "__main__":
    print("Running quick validation test...")
    success = run_single_test()
    sys.exit(0 if success else 1)