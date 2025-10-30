# Python bindings for [`ggml`](https://github.com/ggerganov/ggml)

[![PyPI](https://img.shields.io/pypi/v/ggml-py)](https://pypi.org/project/ggml-py/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ggml-py)](https://pypi.org/project/ggml-py/)
[![PyPI - License](https://img.shields.io/pypi/l/ggml-py)](https://pypi.org/project/ggml-py/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/ggml-py)](https://pypi.org/project/ggml-py/)

> ℹ️ Note this is a fork of the [original project](https://github.com/abetlen/ggml-python) which hasn't been updated July 2024. There have been a number of CVE security patches for `ggml / llama.cpp` since that date. This fork updates `ggml` to the latest. [See the llama.cpp security page for more on this](https://github.com/ggml-org/llama.cpp/security).

Python bindings for the [`ggml`](https://github.com/ggerganov/ggml) tensor library for machine learning.

# Installation

Requirements
- Python 3.10+
- C compiler (gcc, clang, msvc, etc)

You can install `ggml-py` using `pip`:

```bash
pip install ggml-py
```

This will compile ggml using cmake which requires a c compiler installed on your system.
To build ggml with specific features (ie. OpenBLAS, GPU Support, etc) you can pass specific cmake options through the `cmake.args` pip install configuration setting. For example to install ggml-py with cuBLAS support you can run:

```bash
pip install --upgrade pip
pip install ggml-py --config-settings=cmake.args='-DGGML_CUDA=ON'
```

## Options

| Option | Description | Default |
| --- | --- | --- |
| `GGML_CUDA` | Enable cuBLAS support | `OFF` |
| `GGML_CLBLAST` | Enable CLBlast support | `OFF` |
| `GGML_OPENBLAS` | Enable OpenBLAS support | `OFF` |
| `GGML_METAL` | Enable Metal support | `OFF` |
| `GGML_RPC` | Enable RPC support | `OFF` |

# Usage

```python
import ggml
import ctypes

# Allocate a new context with 16 MB of memory
params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024, mem_buffer=None)
ctx = ggml.ggml_init(params)

# Instantiate tensors
x = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)
a = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)
b = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)

# Use ggml operations to build a computational graph
x2 = ggml.ggml_mul(ctx, x, x)
f = ggml.ggml_add(ctx, ggml.ggml_mul(ctx, a, x2), b)

gf = ggml.ggml_new_graph(ctx)
ggml.ggml_build_forward_expand(gf, f)

# Set the input values
ggml.ggml_set_f32(x, 2.0)
ggml.ggml_set_f32(a, 3.0)
ggml.ggml_set_f32(b, 4.0)

# Compute the graph
ggml.ggml_graph_compute_with_ctx(ctx, gf, 1)

# Get the output value
output = ggml.ggml_get_f32_1d(f, 0)
assert output == 16.0

# Free the context
ggml.ggml_free(ctx)
```

# Troubleshooting

If you are having trouble installing `ggml-py` or activating specific features please try to install it with the `--verbose` and `--no-cache-dir` flags to get more information about any issues:

```bash
pip install ggml-py --verbose --no-cache-dir --force-reinstall --upgrade
```

# License

This project is licensed under the terms of the MIT license.
