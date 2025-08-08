
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
