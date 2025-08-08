
#include <iostream>
#include <vector>
#include <fstream>
#include <random>

#include "rnn_performance_test.h"

int main() {
    std::cout << "=== Debug Bidirectional RNN Output Format ===" << std::endl;
    
    // Create test configuration for TEST#5
    TestConfig config;
    config.operator_type = "RNN";
    config.direction = "bidirectional";
    config.activation = "Tanh";
    config.seq_length = 4;
    config.batch_size = 2;
    config.input_size = 8;
    config.hidden_size = 16;
    config.include_sequence_lens = false;
    config.include_initial_h = false;
    config.include_initial_c = false;
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Operator: " << config.operator_type << std::endl;
    std::cout << "  Direction: " << config.direction << std::endl;
    std::cout << "  Shape: seq=" << config.seq_length << ", batch=" << config.batch_size 
              << ", input=" << config.input_size << ", hidden=" << config.hidden_size << std::endl;
    std::cout << std::endl;
    
    // Create cuDNN test
    cudnnDirectionMode_t direction = CUDNN_BIDIRECTIONAL;
    auto cudnn_test = std::make_unique<RNNTest>(config.activation, direction);
    
    if (!cudnn_test->Initialize()) {
        std::cout << "Failed to initialize cuDNN test" << std::endl;
        return 1;
    }
    
    // Generate test data
    int num_directions = 2;
    int num_gates = 1;  // RNN
    
    size_t X_size = config.seq_length * config.batch_size * config.input_size;
    size_t W_size = num_directions * num_gates * config.hidden_size * config.input_size;
    size_t R_size = num_directions * num_gates * config.hidden_size * config.hidden_size;
    size_t B_size = num_directions * 2 * num_gates * config.hidden_size;
    size_t Y_size = config.seq_length * config.batch_size * config.hidden_size * num_directions;
    size_t Y_h_size = num_directions * config.batch_size * config.hidden_size;
    
    std::vector<float> X(X_size);
    std::vector<float> W(W_size);
    std::vector<float> R(R_size);
    std::vector<float> B(B_size);
    std::vector<float> Y_cudnn(Y_size);
    std::vector<float> Y_h_cudnn(Y_h_size);
    
    // Initialize with simple values
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.1f);
    
    for (size_t i = 0; i < X_size; ++i) X[i] = dist(gen);
    for (size_t i = 0; i < W_size; ++i) W[i] = dist(gen);
    for (size_t i = 0; i < R_size; ++i) R[i] = dist(gen);
    for (size_t i = 0; i < B_size; ++i) B[i] = dist(gen);
    
    std::cout << "Running cuDNN forward pass..." << std::endl;
    
    // Run cuDNN forward
    bool success = cudnn_test->Forward(X.data(), W.data(), R.data(), B.data(),
                                     Y_cudnn.data(), Y_h_cudnn.data(), nullptr,
                                     config.seq_length, config.batch_size, 
                                     config.input_size, config.hidden_size);
    
    if (!success) {
        std::cout << "cuDNN forward pass failed" << std::endl;
        return 1;
    }
    
    std::cout << "cuDNN forward pass completed" << std::endl;
    std::cout << std::endl;
    
    // Output shape information
    std::cout << "Output Shapes:" << std::endl;
    std::cout << "Y (main output): [" << config.seq_length << ", " << num_directions 
              << ", " << config.batch_size << ", " << config.hidden_size << "]" << std::endl;
    std::cout << "Y_h (hidden state): [" << num_directions << ", " << config.batch_size 
              << ", " << config.hidden_size << "]" << std::endl;
    std::cout << std::endl;
    
    // Save outputs for analysis
    std::ofstream y_file("cudnn_Y_output.txt");
    std::ofstream yh_file("cudnn_Yh_output.txt");
    
    if (y_file.is_open()) {
        y_file << "# cuDNN Y output format analysis\n";
        y_file << "# Shape: [" << config.seq_length << ", " << num_directions 
               << ", " << config.batch_size << ", " << config.hidden_size << "]\n";
        y_file << "# Format: [timestep, direction, batch, hidden]\n\n";
        
        for (int t = 0; t < config.seq_length; t++) {
            for (int d = 0; d < num_directions; d++) {
                for (int b = 0; b < config.batch_size; b++) {
                    for (int h = 0; h < config.hidden_size; h++) {
                        int idx = t * num_directions * config.batch_size * config.hidden_size + 
                                 d * config.batch_size * config.hidden_size + 
                                 b * config.hidden_size + h;
                        y_file << "Y[" << t << "," << d << "," << b << "," << h << "] = " 
                               << Y_cudnn[idx] << "\n";
                    }
                }
            }
        }
        y_file.close();
        std::cout << "Y output saved to cudnn_Y_output.txt" << std::endl;
    }
    
    if (yh_file.is_open()) {
        yh_file << "# cuDNN Y_h output format analysis\n";
        yh_file << "# Shape: [" << num_directions << ", " << config.batch_size 
               << ", " << config.hidden_size << "]\n";
        yh_file << "# Format: [direction, batch, hidden]\n\n";
        
        for (int d = 0; d < num_directions; d++) {
            for (int b = 0; b < config.batch_size; b++) {
                for (int h = 0; h < config.hidden_size; h++) {
                    int idx = d * config.batch_size * config.hidden_size + 
                             b * config.hidden_size + h;
                    yh_file << "Y_h[" << d << "," << b << "," << h << "] = " 
                            << Y_h_cudnn[idx] << "\n";
                }
            }
        }
        yh_file.close();
        std::cout << "Y_h output saved to cudnn_Yh_output.txt" << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "Debug data saved. Check the output files for format analysis." << std::endl;
    
    return 0;
}
