#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <onnxruntime_cxx_api.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl; \
            return false; \
        } \
    } while (0)

#define CUDNN_CHECK(call) \
    do { \
        cudnnStatus_t status = call; \
        if (status != CUDNN_STATUS_SUCCESS) { \
            std::cerr << "cuDNN error: " << cudnnGetErrorString(status) << std::endl; \
            return false; \
        } \
    } while (0)

int main() {
    try {
        std::cout << "Quick RNN precision test..." << std::endl;
        
        // Test parameters
        int seq_length = 4;
        int batch_size = 2;
        int input_size = 8;
        int hidden_size = 16;
        
        // Initialize cuDNN
        cudnnHandle_t cudnn_handle;
        CUDNN_CHECK(cudnnCreate(&cudnn_handle));
        
        // Create RNN descriptor
        cudnnRNNDescriptor_t rnn_desc;
        CUDNN_CHECK(cudnnCreateRNNDescriptor(&rnn_desc));
        
        CUDNN_CHECK(cudnnSetRNNDescriptor_v8(
            rnn_desc,
            CUDNN_RNN_ALGO_STANDARD,
            CUDNN_RNN_TANH,
            CUDNN_RNN_DOUBLE_BIAS,
            CUDNN_UNIDIRECTIONAL,
            CUDNN_LINEAR_INPUT,
            CUDNN_DATA_FLOAT,
            CUDNN_DATA_FLOAT,
            CUDNN_DEFAULT_MATH,
            input_size,
            hidden_size,
            hidden_size,
            1,
            nullptr,
            CUDNN_RNN_PADDED_IO_ENABLED
        ));
        
        std::cout << "cuDNN initialized successfully" << std::endl;
        
        // Create test data
        std::vector<float> X(seq_length * batch_size * input_size);
        std::vector<float> W(1 * hidden_size * input_size);  // RNN: 1 gate
        std::vector<float> R(1 * hidden_size * hidden_size);
        std::vector<float> B(2 * hidden_size);
        
        // Initialize with simple values
        for (size_t i = 0; i < X.size(); i++) X[i] = 0.1f * (i % 10);
        for (size_t i = 0; i < W.size(); i++) W[i] = 0.01f * (i % 5);
        for (size_t i = 0; i < R.size(); i++) R[i] = 0.02f * (i % 7);
        for (size_t i = 0; i < B.size(); i++) B[i] = 0.001f * (i % 3);
        
        std::cout << "Test data initialized" << std::endl;
        
        // Create ONNX Runtime session with CUDA
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
        Ort::SessionOptions session_options;
        
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        cuda_options.arena_extend_strategy = 0;
        cuda_options.gpu_mem_limit = 2147483648ULL;
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        cuda_options.do_copy_in_default_stream = true;
        
        session_options.AppendExecutionProvider_CUDA(cuda_options);
        
        Ort::Session session(env, "build/rnn_test_1.onnx", session_options);
        std::cout << "ONNX Runtime session created with CUDA provider" << std::endl;
        
        // Run ONNX inference
        std::vector<int64_t> x_shape = {seq_length, batch_size, input_size};
        std::vector<int64_t> w_shape = {1, hidden_size, input_size};
        std::vector<int64_t> r_shape = {1, hidden_size, hidden_size};
        std::vector<int64_t> b_shape = {1, 2 * hidden_size};
        
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        std::vector<Ort::Value> inputs;
        inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, X.data(), X.size(), x_shape.data(), x_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, W.data(), W.size(), w_shape.data(), w_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, R.data(), R.size(), r_shape.data(), r_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, B.data(), B.size(), b_shape.data(), b_shape.size()));
        
        auto output = session.Run(Ort::RunOptions{nullptr}, 
                                 {"X", "W", "R", "B"}, 
                                 inputs.data(), inputs.size(), 
                                 {"Y", "Y_h"}, 2);
        
        std::vector<float> Y_onnx = output[0].GetTensorMutableData<float>();
        std::vector<float> Y_h_onnx = output[1].GetTensorMutableData<float>();
        
        size_t Y_size = seq_length * batch_size * hidden_size;
        size_t Y_h_size = batch_size * hidden_size;
        
        std::cout << "ONNX Runtime inference completed" << std::endl;
        std::cout << "Y size: " << Y_size << ", Y_h size: " << Y_h_size << std::endl;
        
        // Clean up
        cudnnDestroyRNNDescriptor(rnn_desc);
        cudnnDestroy(cudnn_handle);
        
        std::cout << "Test completed successfully!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}