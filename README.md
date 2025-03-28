# Xavier - A Deep Learning Framework

## Overview
Xavier is a lightweight deep learning framework designed to provide Metal-accelerated tensor operations similar to PyTorch. Future releases will include CUDA support for NVIDIA GPUs.

## Requirements
- Python 3.12+
- C++ 23
- NumPy
- Mypy (type hints and docs)
- Pybind 11 (C++ bindings)
- Metal-capable device (macOS)

## Installation
1. Clone the repository
```bash
git clone https://github.com/yourusername/xavier.git
cd xavier
```

2. Build the project
```bash
cd xavier/core
cmake -S . -B build
cmake --build build
```
* CMake will generate a `.so` file.
* Place this file inside `python` directory.
* Run `stubgen -m xavier` to enable autocomplete.


## Usage
1. Import Xavier in your Python code:
```python
import python.xavier as xv
from python.xavier import MTLGraph, Array, MTLContext
```

2. Initialize Metal context and create tensors:
```python
ctx = MTLContext("./xavier/build/backend/metal/kernels.metallib")
shape = [2, 3, 4]
x1 = Array.from_numpy(np.random.randn(*shape).astype(np.float32))
x2 = Array.from_numpy(np.random.randn(*shape).astype(np.float32))
x3 = Array.full(shape, 2.0, dtype=np.float32)
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
- Supported operations:
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
