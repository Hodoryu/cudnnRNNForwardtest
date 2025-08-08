# cuDNN RNN Performance Test

This is a comprehensive performance and functionality test suite for cuDNN's RNN forward interface (`cudnnRNNForward`). The test covers RNN, GRU, and LSTM algorithms with various configurations and shapes.

## Features

- **Comprehensive Testing**: Tests RNN, GRU, and LSTM algorithms
- **Multiple Shapes**: Covers small, medium, and large input shapes
- **Performance Measurement**: Accurate timing and GFLOPS calculation
- **500+ Test Cases**: Extensive microbenchmark coverage
- **Error Handling**: Robust error checking and reporting
- **Output Verification**: Validates output for NaN/Inf values

## Test Coverage

### Algorithms
- RNN with ReLU activation
- RNN with Tanh activation
- GRU (Gated Recurrent Unit)
- LSTM (Long Short-Term Memory)

### Shapes
- **Small**: Batch sizes 1-8, sequence lengths 10-100, input sizes 32-256
- **Medium**: Batch sizes 16-64, sequence lengths 128-512, input sizes 512-1024
- **Large**: Batch sizes 128-256, sequence lengths 1024-2048, input sizes 2048-4096

### Configurations
- Unidirectional and bidirectional RNNs
- 1-8 layers
- Different input/hidden size ratios
- Edge cases and performance-oriented configurations

## Build Requirements

- NVIDIA CUDA Toolkit (11.0 or later)
- cuDNN Library (8.0 or later)
- GCC compiler with C++17 support
- Linux operating system

## Building

```bash
# Create build directory
mkdir build && cd build

# Check dependencies
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the test
cmake --build .

# Build with debug symbols
cmake .. -DCMAKE_BUILD_TYPE=Debug && cmake --build .

# Build with profiling support
cmake .. -DCMAKE_BUILD_TYPE=Profile && cmake --build .

# Run tests
ctest
```

## Running the Test

```bash
# Run the test
cd build
./rnn_performance_test

# Or with CTest
ctest --verbose
```

## Environment Variables

You can customize the build with these environment variables:

```bash
# Set CUDA installation path
export CUDA_HOME=/usr/local/cuda

# Set cuDNN installation path
export CUDNN_HOME=/usr/local/cudnn
```

Or use CMake cache variables:

```bash
# Configure with custom paths
cmake .. -DCUDA_HOME=/path/to/cuda -DCUDNN_HOME=/path/to/cudnn
```

## Output

The test produces detailed output including:

1. **System Information**
   - cuDNN version
   - CUDA runtime version
   - Device properties

2. **Test Execution**
   - Individual test results
   - Execution time (milliseconds)
   - Performance (GFLOPS)
   - Output verification status

3. **Summary Statistics**
   - Total tests run
   - Pass/fail count
   - Success rate
   - Average performance
   - Performance breakdown by algorithm

## Sample Output

```
cuDNN RNN Performance and Functionality Test
=============================================
cuDNN Version: 8900
CUDA Runtime Version: 12000
Device: NVIDIA GeForce RTX 4090
Compute Capability: 8.9

Generated 540 test configurations

Test 1/540: RNN_RELU BS=1 SL=10 IS=32 HS=32 NL=1 BI=No
  Time: 0.123 ms
  Performance: 0.85 GFLOPS
  Output verification: PASSED

Test 2/540: RNN_RELU BS=1 SL=10 IS=32 HS=32 NL=2 BI=No
  Time: 0.145 ms
  Performance: 1.02 GFLOPS
  Output verification: PASSED

...

Test Summary
============
Total tests: 540
Passed: 540
Failed: 0
Success rate: 100.00%
Average performance: 125.67 GFLOPS

Performance by RNN Mode
=======================
RNN_RELU: Avg=145.23 GFLOPS, Min=0.85 GFLOPS, Max=892.34 GFLOPS
RNN_TANH: Avg=138.91 GFLOPS, Min=0.92 GFLOPS, Max=856.78 GFLOPS
GRU: Avg=112.45 GFLOPS, Min=0.78 GFLOPS, Max=723.56 GFLOPS
LSTM: Avg=98.76 GFLOPS, Min=0.65 GFLOPS, Max=645.23 GFLOPS

Test completed successfully!
```

## Test Details

### Performance Measurement
- Uses high-resolution timers for accurate measurement
- Includes warm-up iterations to account for GPU initialization
- Reports average time over multiple iterations
- Calculates GFLOPS based on theoretical operation counts

### Functionality Testing
- Verifies output tensors for NaN/Inf values
- Tests with different RNN configurations
- Includes edge cases and stress tests

### Memory Management
- Proper CUDA memory allocation and deallocation
- Workspace and reserve space management
- Error handling for out-of-memory conditions

## Error Handling

The test includes comprehensive error handling:
- CUDA runtime errors
- cuDNN API errors
- Memory allocation failures
- Invalid configuration detection

## Customization

You can modify the test configurations in the `generate_test_configs()` function to:
- Add more test cases
- Adjust shape ranges
- Change algorithm parameters
- Modify iteration counts

## Troubleshooting

### Common Issues

1. **CUDA/cuDNN not found**
   - Ensure CUDA_HOME and CUDNN_HOME are set correctly
   - Verify library paths in CMake configuration
   - Check CMake output for dependency information

2. **Out of memory**
   - Reduce batch sizes or sequence lengths
   - Test on a GPU with more memory

3. **Compilation errors**
   - Check GCC version (requires C++17 support)
   - Verify CUDA/cuDNN installation
   - Ensure CMake version >= 3.18

### Debug Mode

For debugging issues, build with debug symbols:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug && cmake --build .
gdb ./build/rnn_performance_test
```

### CMake Configuration

Additional CMake options:

```bash
# Set build type
cmake .. -DCMAKE_BUILD_TYPE=Release

# Set cuDNN path explicitly
cmake .. -DCUDNN_HOME=/path/to/cudnn

# Clean build
rm -rf build && mkdir build && cmake .. && cmake --build .
```

## Performance Notes

- Performance varies significantly based on GPU architecture
- Larger batch sizes generally yield better performance
- Bidirectional RNNs are approximately 2x slower than unidirectional
- LSTM is typically the slowest due to more complex operations
- Performance is measured in GFLOPS (billions of floating-point operations per second)

## License

This test suite is provided as-is for performance evaluation purposes.