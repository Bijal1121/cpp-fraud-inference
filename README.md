# C++ Fraud Detection Inference System

A lightweight machine learning inference project that trains a fraud-detection model in Python, exports it to ONNX, and runs transaction risk scoring using a C++17 inference pipeline with ONNX Runtime.

This project demonstrates how a trained ML model can be moved from a Python training workflow into a production-style C++ inference environment.

---

## Project Overview

The goal of this project is to build a small but practical fraud-detection inference system using C++.

Instead of only training a model in Python, this project focuses on the deployment side of machine learning. A fraud classification model is trained in Python, converted into the ONNX format, and then loaded inside a C++ application for local CPU inference.

The C++ application runs transaction risk scoring, returns the predicted class, outputs fraud probability, and measures inference latency using `std::chrono`.

---

## Why This Project

Many AI Engineer and Machine Learning Engineer roles require not only model-building knowledge, but also the ability to understand inference, latency, deployment, and production runtime behavior.

This project highlights:

* C++17 programming for ML inference
* ONNX model execution using ONNX Runtime
* Python-to-C++ model deployment workflow
* Local CPU-based transaction risk scoring
* Inference latency benchmarking
* CMake-based C++ build configuration

---

## Tech Stack

| Category              | Tools / Technologies                   |
| --------------------- | -------------------------------------- |
| Programming Languages | C++17, Python                          |
| ML Frameworks         | scikit-learn, ONNX, ONNX Runtime       |
| Model Export          | skl2onnx                               |
| Build System          | CMake                                  |
| Compiler              | MSVC / Visual Studio Build Tools       |
| Package Management    | NuGet                                  |
| IDE                   | Visual Studio Code                     |
| Benchmarking          | `std::chrono`, Python time measurement |

---

## Project Structure

```text
C++ fraud detection/
│
├── data/
│   └── sample_transaction.csv
│
├── models/
│   └── fraud_model.onnx
│
├── python/
│   ├── train_model.py
│   └── benchmark_python.py
│
├── src/
│   └── main.cpp
│
├── packages/
│   └── Microsoft.ML.OnnxRuntime.1.27.0/
│
├── build/
│
├── CMakeLists.txt
└── README.md
```

---

## Features

* Trains a fraud-detection classifier using Python
* Exports the trained model to ONNX format
* Loads the ONNX model inside a C++17 application
* Runs local CPU inference using ONNX Runtime
* Outputs prediction label and fraud probability
* Measures C++ inference latency
* Compares C++ ONNX Runtime inference with Python ONNX Runtime inference
* Uses CMake for building the C++ executable

---

## Model Workflow

```text
Synthetic Transaction Data
        ↓
Python Model Training
        ↓
ONNX Model Export
        ↓
C++ ONNX Runtime Inference
        ↓
Fraud Probability + Latency Output
```

---

## How It Works

### 1. Train the Model in Python

A fraud-detection classifier is trained using Python and scikit-learn. The model learns to classify a transaction as either:

```text
0 = Not Fraud
1 = Fraud
```

After training, the model is exported into ONNX format so it can be used outside Python.

---

### 2. Export the Model to ONNX

ONNX allows the trained model to be used across different programming languages and runtime environments.

The exported model is saved as:

```text
models/fraud_model.onnx
```

---

### 3. Load the Model in C++

The C++ application uses ONNX Runtime to load the ONNX model file.

The program creates an inference session, prepares transaction input features, runs the model, and receives prediction outputs.

---

### 4. Run Fraud Inference

The C++ program sends transaction features into the ONNX model and receives two outputs:

```text
label
probabilities
```

The label shows the predicted class.

The probabilities show the model confidence for each class.

---

### 5. Measure Inference Latency

The C++ program uses `std::chrono` to measure how long the inference step takes.

This helps evaluate how fast the model can make predictions in a local CPU environment.

---

## Sample C++ Output

```text
Input name: input
Output 0: label
Output 1: probabilities
Prediction label: 1
Not fraud probability: 0.0467047
Fraud probability: 0.953295
C++ inference latency: 0.2035 ms
```

---

## Benchmark Results

| Runtime             |                   Latency |
| ------------------- | ------------------------: |
| C++ ONNX Runtime    |                 0.2035 ms |
| Python ONNX Runtime | 0.0139 ms per transaction |

The benchmark compares local CPU inference using the same ONNX model.

Python ONNX Runtime internally calls optimized native ONNX Runtime execution, so the Python version is not pure Python execution. Both versions use ONNX Runtime as the backend.

The C++ result demonstrates that the model can be deployed and executed successfully in a C++ runtime environment with sub-millisecond local CPU inference.

---

## Key Learning Outcomes

Through this project, I learned how to:

* Train a simple machine learning model in Python
* Export a trained model to ONNX format
* Use ONNX Runtime from C++
* Configure a C++ project using CMake
* Link native libraries in a Windows C++ build
* Handle model inputs and outputs in C++
* Measure inference latency using `std::chrono`
* Compare runtime behavior between Python and C++

---

## Setup Instructions

### Prerequisites

Install the following tools:

* Python 3.x
* Visual Studio Code
* Visual Studio Build Tools / Visual Studio Community
* CMake
* NuGet
* ONNX Runtime

---

## Python Setup

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate the virtual environment:

```powershell
.\.venv\Scripts\activate
```

Install Python dependencies:

```powershell
pip install numpy pandas scikit-learn skl2onnx onnxruntime
```

---

## Train and Export the Model

Run the training script:

```powershell
python python/train_model.py
```

This creates the ONNX model:

```text
models/fraud_model.onnx
```

---

## Install ONNX Runtime for C++

This project uses the NuGet version of ONNX Runtime for Windows.

Download `nuget.exe`:

```powershell
Invoke-WebRequest -Uri "https://dist.nuget.org/win-x86-commandline/latest/nuget.exe" -OutFile ".\nuget.exe"
```

Install ONNX Runtime:

```powershell
.\nuget.exe install Microsoft.ML.OnnxRuntime -Version 1.27.0 -OutputDirectory packages
```

After installation, the ONNX Runtime package should be located here:

```text
packages/Microsoft.ML.OnnxRuntime.1.27.0/
```

---

## Build the C++ Project

Configure the project using CMake:

```powershell
cmake -S . -B build -DONNXRUNTIME_DIR="<path-to-Microsoft.ML.OnnxRuntime.1.27.0>"
```

Example on Windows:

```powershell
cmake -S . -B build -DONNXRUNTIME_DIR="C:\Users\Bijal\Desktop\Bijal\Projects\C++ fraud detection\packages\Microsoft.ML.OnnxRuntime.1.27.0"
```

Build the project:

```powershell
cmake --build build --config Release
```

---

## Run the C++ Inference Program

Run the executable:

```powershell
.\build\bin\Release\fraud_infer.exe
```

Expected output:

```text
Input name: input
Output 0: label
Output 1: probabilities
Prediction label: 1
Not fraud probability: 0.0467047
Fraud probability: 0.953295
C++ inference latency: 0.2035 ms
```

---

## Run Python Benchmark

Run the Python benchmark script:

```powershell
python python/benchmark_python.py
```

Sample output:

```text
Python ONNX Runtime average latency: 0.0139 ms per transaction
Input: input
Outputs: ['label', 'probabilities']
```

---

## CMake Configuration

This project uses the following `CMakeLists.txt` setup:

```cmake
cmake_minimum_required(VERSION 3.15)
project(FraudInfer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ONNXRUNTIME_DIR)
    message(FATAL_ERROR "Please set ONNXRUNTIME_DIR")
endif()

include_directories("${ONNXRUNTIME_DIR}/build/native/include")

add_executable(fraud_infer src/main.cpp)

target_link_libraries(
    fraud_infer
    "${ONNXRUNTIME_DIR}/runtimes/win-x64/native/onnxruntime.lib"
)

set_target_properties(fraud_infer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_custom_command(TARGET fraud_infer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${ONNXRUNTIME_DIR}/runtimes/win-x64/native/onnxruntime.dll"
    $<TARGET_FILE_DIR:fraud_infer>
)
```

---

## Resume Description

**C++ Fraud Detection Inference System** | C++, ONNX Runtime, Python, CMake

* Built a C++17 fraud-detection inference pipeline by exporting a Python-trained model to ONNX for local CPU execution.
* Implemented ONNX Runtime model loading, feature preprocessing, probability-based classification, and transaction risk scoring.
* Benchmarked local CPU inference latency with `std::chrono`, achieving 0.20 ms C++ inference and comparing against Python ONNX Runtime.

---

## Future Improvements

Possible future upgrades:

* Add batch inference for multiple transactions
* Add CSV file input for real transaction rows
* Improve C++ benchmarking with warm-up runs and average latency
* Add a REST API using a lightweight C++ framework
* Add Docker support
* Test with a real fraud-detection dataset
* Add logging and better error handling
* Add GitHub Actions for automated build testing

---

## Project Status

Completed:

* Python model training
* ONNX model export
* C++ ONNX Runtime inference
* CMake build
* Local CPU latency benchmark
* Python vs. C++ runtime comparison
