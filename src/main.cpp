#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
std::vector<float> read_first_transaction(const std::string& csv_path) {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
    throw std::runtime_error("Could not open CSV file: " + csv_path);
    }
    std::string header;
    std::getline(file, header); // skip column names
    std::string line;
    std::getline(file, line); // read first data row
    std::stringstream ss(line);
    std::string cell;
    std::vector<float> features;
    // CSV has 10 features + label. We only read the first 10 columns.
    int col = 0;
    while (std::getline(ss, cell, ',') && col < 10) {
    features.push_back(std::stof(cell));
    col++;
    }
    if (features.size() != 10) {
    throw std::runtime_error("Expected 10 features, got " + std::to_string(features.size()));
    }
    return features;
}

int main() {
    try {
        const std::string model_path = "models/fraud_model.onnx";
        const std::string csv_path = "data/sample_transactions.csv";
        std::vector<float> input_values = read_first_transaction(csv_path);
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "fraud_inference");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

#ifdef _WIN32
        std::wstring w_model_path(model_path.begin(), model_path.end());
        Ort::Session session(env, w_model_path.c_str(), session_options);
#else
        Ort::Session session(env, model_path.c_str(), session_options);
#endif
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name = session.GetInputNameAllocated(0, allocator);
        auto output_name0 = session.GetOutputNameAllocated(0, allocator);
        auto output_name1 = session.GetOutputNameAllocated(1, allocator);
        std::cout << "Input name: " << input_name.get() << std::endl;
        std::cout << "Output 0: " << output_name0.get() << std::endl;
        std::cout << "Output 1: " << output_name1.get() << std::endl;
        std::vector<int64_t> input_shape = {1, 10};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault
        );

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_values.data(),
            input_values.size(),
            input_shape.data(),
            input_shape.size()
        );

        const char* input_names[] = {input_name.get()};
        const char* output_names[] = {output_name0.get(), output_name1.get()};

        auto start = std::chrono::high_resolution_clock::now();
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            2
        );

        auto end = std::chrono::high_resolution_clock::now();

        double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();

        // For this sklearn ONNX export: output 0 = label, output 1 = probability tensor.
        int64_t* label = output_tensors[0].GetTensorMutableData<int64_t>();
        float* probabilities = output_tensors[1].GetTensorMutableData<float>();
        float not_fraud_prob = probabilities[0];
        float fraud_prob = probabilities[1];

        std::cout << "Prediction label: " << label[0] << std::endl;
        std::cout << "Not fraud probability: " << not_fraud_prob << std::endl;
        std::cout << "Fraud probability: " << fraud_prob << std::endl;
        std::cout << "C++ inference latency: " << latency_ms << " ms" << std::endl;

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
