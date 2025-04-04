# Xavier - A Deep Learning Framework

## Overview
Xavier is a lightweight deep learning framework designed to provide Metal-accelerated tensor operations similar to PyTorch. Future releases will hopefully include CUDA support for NVIDIA GPUs.

## Requirements
A virtual environment(e.g., Conda) is recommended before installing Python packages:
- Python 3.12+
- C++ 23
- NumPy (>=2.0 but should work for any version)
- Mypy >= 1.15 (type hints and docs)
- Pybind11 >= 2.13.6 (C++ bindings)
- Metal 3.2 and metal-cpp (macOS)
- Pytest >= 8.3.4 (optional, mainly for testing)

## Installation
1. Clone the repository
```bash
git clone https://github.com/yourusername/xavier.git
cd xavier
```

2. Build the project
* Do the following on macOS:
  * Download `metal-cpp` using the link https://developer.apple.com/metal/cpp/.
  * Move the metal-cpp into `xavier` directory.
* Edit the system path to python and pybind11 in `xavier/CMakeLists.txt`.
* Build the project and generate `.so` file using the following commands:
```bash
cd xavier/core
cmake -S . -B build
cmake --build build
```
* Place the generated `.so` file inside `python` directory or anywhere else you'd like, treat the `.so` file as a Python module.
* Place the generated `.metallib` file by Metal anywhere you'd like, use that path to initialize `MTLContext` to run on Macos Metal GPU.
* Run `stubgen -m xavier -o .` after installing mypy to enable autocomplete and type hints.


## Usage
1. Import Xavier in your Python code:
```python
# Note: use the path to `.so` module
# The path to `.so` module in the example is python/xavier.cpython-312-darwin.so
import python.xavier as xv
from python.xavier import MTLGraph, Array, MTLContext, f32
```

2. Initialize Metal context and create tensors:
```python
ctx = MTLContext("path to your generated metallib file")
shape = [2, 3, 4]
x1 = Array.from_numpy(np.random.randn(*shape).astype(np.float32))
x2 = Array.from_numpy(np.random.randn(*shape).astype(np.float32))
x3 = Array.full(shape, 2.0, dtype=f32)
```

3. Define computations and run:
```python
# Define computation graph
out = ((x1 + x2) * x3).exp()

# Create and compile graph
g = MTLGraph(out, ctx)
g.compile()

# Execute forward and backward passes
g.forward()
g.backward()
```

## Features
- Metal-accelerated tensor operations
- Automatic differentiation
- Full computational graph forward and backward propagation
- Well supported operations:
  - Initialization operations: full, arange, ones, zeros, from_numpy, to_numpy, from_buffer
  - Common tensor operations: reshape, permute, matmul, slice, transpose
  - Element-wise operations: add, sub, mul, div, exp, log, neg(negation), recip(reciprocal), sqrt, sq(square)
- NumPy integration
- Backprop is not fully supported for some operations, namely slice, broadcast, and reduction operations

## Examples
Check the `tests` directory for example implementations of:
- Tensor operations
- Gradient calculations
- Metal backend usage
- PyTorch comparison tests

## Acknowledgement
These resources inspired me to do the project:
- MLX from Apple (https://github.com/ml-explore/mlx)
- PyTorch from Meta (https://github.com/pytorch/pytorch)
- Tinygrad by George Hotz (https://github.com/tinygrad/tinygrad)
- Micrograd by Karpathy (https://github.com/karpathy/micrograd)
