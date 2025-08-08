# RNN Performance Test - CUDA Execution Provider Setup

## Issue
The RNN performance test is currently falling back to CPU Execution Provider because the CUDA provider library is missing:

```
Failed to load library libonnxruntime_providers_shared.so: cannot open shared object file: No such file or directory
```

## Solution
To enable GPU acceleration with ONNX Runtime, you need to install the CUDA-enabled version of ONNX Runtime.

### Installation Steps

1. **Install CUDA-enabled ONNX Runtime**:
   ```bash
   # For Ubuntu/Debian:
   pip install onnxruntime-gpu
   
   # Or download from official releases:
   # https://github.com/microsoft/onnxruntime/releases
   ```

2. **Verify CUDA provider library**:
   ```bash
   # Check if the library exists
   find /usr -name "libonnxruntime_providers_shared.so" 2>/dev/null
   # Or in Python environment
   python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
   ```

3. **Expected output should include**:
   - `['CUDAExecutionProvider', 'CPUExecutionProvider']`

## Current Status
- ✅ Test functionality is working correctly
- ✅ ONNX model creation is working
- ✅ Precision validation is working
- ✅ Performance metrics are being collected
- ⚠️ Using CPU Execution Provider (fallback)
- 📋 Need CUDA-enabled ONNX Runtime for GPU acceleration

## Impact
- Performance comparisons between cuDNN and ONNX Runtime are not accurate
- ONNX Runtime is running on CPU while cuDNN is running on GPU
- This affects speedup calculations and throughput metrics

## Verification
After installing the CUDA provider, the test should show:
```
Model loaded with CUDA Execution Provider
```

Instead of:
```
Model loaded with CPU Execution Provider
```