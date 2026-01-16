#
**MLP-MLIR** – An MLIR-based Compiler Infrastructure for Neural Networks

![MLIR](https://img.shields.io/badge/MLIR-LLVM-blue)
![C++](https://img.shields.io/badge/C%2B%2B-17-green)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange)

---

## Overview

**MLP-MLIR** is an experimental compiler framework built on LLVM MLIR that focuses on neural network representation, transformation, and lowering. The project currently targets **Multi-Layer Perceptrons (MLPs)** and provides a foundation for future Transformer and LLM support.

The current implementation targets **Multi-Layer Perceptrons (MLPs)** and demonstrates:

- Programmatic construction of MLIR IR in C++
- Custom dialect operations for neural network primitives
- Tensor-level computation and inspection
- A standalone MLIR execution pipeline

The project is designed with a **long-term roadmap** toward **Transformers, Large Language Models (LLMs), and heterogeneous hardware backends**.

---

## Goals

- ✅ Provide a clean MLIR-based representation for **MLP workloads**
- ✅ Experiment with **custom neural network dialect extensions**
- 🔜 Extend to **Transformer architectures**
- 🔜 Support **LLM-scale graphs**
- 🔜 Enable **multi-hardware lowering** (CPU, GPU, accelerators)
- 🔜 Explore **optimization passes and graph-level rewrites**

---

## Why MLIR?

MLIR provides:
- Multi-level abstraction (tensor → linalg → LLVM)
- Dialect extensibility
- Hardware-agnostic IR design
- First-class support for compiler transformations

This makes it ideal for **machine learning compilers** that must evolve across:
- models
- hardware targets
- optimization strategies

---

## Current Capabilities

### Implemented
- MLP-style operations (Add, Mul, ReLU)
- Tensor constants and elementwise ops
- Print operations for debugging
- MLIR IR construction via C++ builders
- Standalone MLIR driver

### In Progress
- Shape-aware ops
- TOSA / Linalg interoperability
- Type consistency (f32 / f64)
- Modular builder separation

---

## Roadmap

### Phase 1 — MLP Foundations (Current)
- Custom ops
- Tensor algebra
- MLIR builder utilities

### Phase 2 — Transformer Support
- Linear layers
- Attention mechanisms
- Softmax, LayerNorm
- Sequence modeling

### Phase 3 — LLM-Scale Compilation
- Graph-level optimizations
- Memory planning
- Operator fusion

### Phase 4 — Multi-Hardware Lowering
- CPU (LLVM)
- GPU (NVVM / ROCm)
- Accelerator targets
- Backend-specific optimization passes
