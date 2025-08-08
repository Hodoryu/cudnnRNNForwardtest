#include <iostream>
#include <vector>
#include <cassert>

// Simple test to verify weight reorganization logic
class MockRNNTest {
public:
    virtual std::vector<int> GetWLinLayerIds() const = 0;
    virtual std::vector<int> GetRLinLayerIds() const = 0;
    virtual int GetNumGates() const = 0;
};

class MockGRUTest : public MockRNNTest {
protected:
    std::vector<int> GetWLinLayerIds() const override {
        return {1, 0, 2};  // ONNX Runtime GRU W layer IDs
    }

    std::vector<int> GetRLinLayerIds() const override {
        return {4, 3, 5};  // ONNX Runtime GRU R layer IDs
    }

    int GetNumGates() const override {
        return 3;
    }
};

class MockLSTMTest : public MockRNNTest {
protected:
    std::vector<int> GetWLinLayerIds() const override {
        return {0, 3, 1, 2};  // ONNX Runtime LSTM W layer IDs
    }

    std::vector<int> GetRLinLayerIds() const override {
        return {4, 7, 5, 6};  // ONNX Runtime LSTM R layer IDs
    }

    int GetNumGates() const override {
        return 4;
    }
};

void test_weight_organization(MockRNNTest* test, const std::string& name) {
    std::cout << "Testing " << name << " weight organization..." << std::endl;
    
    int num_gates = test->GetNumGates();
    std::vector<int> w_layer_ids = test->GetWLinLayerIds();
    std::vector<int> r_layer_ids = test->GetRLinLayerIds();
    
    std::cout << "  W layer IDs: [";
    for (int id : w_layer_ids) std::cout << id << " ";
    std::cout << "]" << std::endl;
    
    std::cout << "  R layer IDs: [";
    for (int id : r_layer_ids) std::cout << id << " ";
    std::cout << "]" << std::endl;
    
    // Test W weight order
    std::cout << "  W weight gate order: ";
    for (int idx = 0; idx < num_gates; idx++) {
        int gate = w_layer_ids[idx] % num_gates;
        std::cout << gate << " ";
    }
    std::cout << std::endl;
    
    // Test R weight order
    std::cout << "  R weight gate order: ";
    for (int idx = 0; idx < num_gates; idx++) {
        int gate = r_layer_ids[idx] % num_gates;
        std::cout << gate << " ";
    }
    std::cout << std::endl;
    
    std::cout << std::endl;
}

int main() {
    std::cout << "Testing weight reorganization fixes..." << std::endl << std::endl;
    
    MockGRUTest gru_test;
    MockLSTMTest lstm_test;
    
    test_weight_organization(&gru_test, "GRU");
    test_weight_organization(&lstm_test, "LSTM");
    
    std::cout << "Weight organization test completed!" << std::endl;
    return 0;
}