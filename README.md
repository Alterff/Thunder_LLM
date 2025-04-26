Thunder_LLM
Overview

Thunder_LLM is a high-performance Large Language Model (LLM) framework designed for real-time use in robotics and embedded systems.
It deploys both the model and tokenizer using ONNX and C++, ensuring low-latency, high-efficiency, and hardware-friendly execution — ideal for demanding robotic environments where every millisecond matters.
Key Features

    🚀 Fast Inference: Model and tokenizer run natively through ONNX runtime in C++ for minimal overhead.

    🛠️ Optimized for Robotics: Designed with real-time and resource-constrained robotic systems in mind.

    🔗 Seamless Integration: Lightweight C++ interface makes it easy to plug into existing robotic control systems.

    ⚡ Tokenizer Deployment: Tokenization is done ONNX-side, eliminating Python dependencies at runtime.

    📦 Cross-Platform: Can be deployed on Linux, Windows, or embedded systems supporting ONNX Runtime.

Architecture

    Tokenizer (ONNX): Converts input text to tokens efficiently.

    LLM (ONNX): Processes tokens to generate intelligent responses or action commands.

    C++ Inference Engine: Orchestrates the full pipeline, handling inputs, outputs, and post-processing.

    Robotics Interface Layer: (Optional) Hooks to connect LLM outputs directly into robotic control systems.
