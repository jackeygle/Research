# Research Portfolio 🔬

This repository contains a collection of my personal research projects spanning **Large Language Model (LLM) Security**, **Robotics**, **Computer Vision**, and **Efficient Deep Learning**.

## Projects Overview

### 1. [LLM Security: Red Teaming & Defenses](llm-security-mini-project/)
A comprehensive evaluation harness for testing the safety of Open-Source LLMs (e.g., Llama 3, Phi-3).
- **Key Features**: Automated prompt injection attacks, multi-agent adversarial simulation (Red Teaming), and lightweight defense layers (input/output filtering).
- **Core Tech**: Hugging Face Transformers, PyTorch, Multi-Agent Orchestration.

### 2. [RoboTwin: Bimanual Manipulation Benchmark](robotwin/)
A large-scale benchmark and data generator for bimanual robotic manipulation tasks.
- **Highlights**: Accepted to **CVPR 2025**. Focuses on domain randomization, digital twins, and robust policy learning.
- **Impact**: Provides 100k+ trajectories and standardizes evaluation for dual-arm robots.

### 3. [Knowledge Distillation](knowledge-distillation/)
Research on neural network compression techniques to transfer knowledge from large "Teacher" models (ResNet-34) to compact "Student" models (MobileNetV2/ResNet-18).
- **Results**: Achieved ~6x compression ratio while maintaining 90% of the teacher's accuracy on CIFAR-10/TinyImageNet.
- **Methods**: Logits-based distillation (KL Divergence), structured pruning.

### 4. [Action Recognition](action-recognition/)
Video understanding system based on **3D ResNets (R3D-18)** for recognizing human actions in video streams.
- **Dataset**: UCF101.
- **Performance**: Optimized for efficient inference on video clips using spatiotemporal convolutions.

### 5. [Adaptive Trajectory Prediction](adaptive-prediction/)
Implementation of meta-learning approaches for behavior prediction in autonomous driving.
- **Focus**: expanding the deployment envelope of prediction models to new environments using adaptive meta-learning.

### 6. [Deep Past Translation](deep-past-translation/)
Research on sequence-to-sequence modeling for long-term time series or trajectory translation tasks.

---

## Technical Stack
- **Deep Learning**: PyTorch, Hugging Face, TensorFlow
- **Robotics**: Isaac Sim, ManiSkill
- **Compute**: Slurm Cluster (A100/V100/H100 optimization)
- **DevOps**: Docker, CI/CD pipelines
