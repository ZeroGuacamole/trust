# trust

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![CI](https://github.com/danite/trust/actions/workflows/integration.yml/badge.svg)

A lightweight, efficient, and pure-Rust implementation of n-dimensional tensors, suitable for numerical computations, machine learning, and more.

### Milestone 1: Core Tensor Operations ✅

- [x] Tensor data structure with:
  - [x] Data storage
  - [x] Shape and stride information
- [x] Element-wise operations:
  - [x] Addition
  - [x] Subtraction
  - [x] Multiplication
  - [x] Division
- [x] Basic tensor manipulation:
  - [x] Reshaping
  - [x] Squeezing
  - [x] Expanding dimensions
- [x] Tensor indexing and slicing.
- [x] Comprehensive tests for the above functionalities.

### Milestone 2: Advanced Tensor Manipulation

- [ ] Broadcasting support for operations between tensors of different shapes.
- [ ] Reduction operations:
  - [ ] Sum
  - [ ] Mean
  - [ ] Max
  - [ ] Min along specified dimensions
- [ ] More advanced tensor operations:
  - [ ] Matrix multiplication
  - [ ] Dot product
  - [ ] Transpose

### Milestone 3: Automatic Differentiation Basics

- [ ] Computation graph structure:
  - [ ] Nodes for operations
  - [ ] Edges for tensor data
- [ ] Basic autograd operations:
  - [ ] Implement forward and backward methods for basic operations (element-wise ops, matrix multiplication)
- [ ] Backpropagation:
  - [ ] Given a final tensor and its gradient, propagate gradients back through the computation graph.

### Milestone 4: Advanced Autograd and Optimizers

- [ ] Advanced autograd:
  - [ ] Handle operations like reductions, advanced indexing, etc.
- [ ] Basic optimizers:
  - [ ] Gradient descent
  - [ ] Stochastic gradient descent
- [ ] Extend the Tensor structure to hold intermediate gradients and computation history.

### Milestone 5: GPU Support & Acceleration

- [ ] Basic GPU tensor representation.
- [ ] GPU kernels for core tensor operations.
- [ ] GPU support for advanced operations and autograd.
- [ ] Efficient CPU-GPU memory transfer mechanisms.

### Milestone 6: Extensions and Utilities

- [ ] Serialization and deserialization of tensors.
- [ ] Custom operations: API for users to define their own tensor operations.
- [ ] Random number generation for tensors.
- [ ] Import/export compatibility with popular tensor libraries or model formats.

### Milestone 7: Optimizations and Performance

- [ ] Memory optimization:
  - [ ] In-place operations
  - [ ] Memory pooling
- [ ] Parallelism and concurrency:
  - [ ] Multi-threaded operations for large tensors
- [ ] Algorithmic optimizations for specific tensor operations.

### Milestone 8: Documentation, Examples, and Community Building

- [ ] Comprehensive API documentation.
- [ ] Tutorials and guides.
- [ ] Sample projects or applications using the library.

### Milestone 9: Robustness & Interoperability

- [ ] Comprehensive test suite, including stress tests and edge cases.
- [ ] Benchmarking against established tensor libraries.
- [ ] Interoperability tools:
  - [ ] C/C++ bindings
  - [ ] ONNX compatibility, etc.
- [ ] Error handling and reporting mechanisms.
