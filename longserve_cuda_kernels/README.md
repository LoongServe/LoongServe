# LongServe CUDA Kernels

This library implements some CUDA kernels required by LongServe.

## Motivation

OpenAI's Triton, a DSL for writing highly efficient CUDA kernels, enables us to
efficiently write custom kernels in pure Python. However, due to the high kernel
launching overhead (usually 100 ~ 400 us), it is not suitable for decoding stage.

To work around this issue, we implement some CUDA kernels in C++/CUDA and wrap them
into Python via PyTorch's C++ API. This way, we can achieve low kernel launching
overhead and high performance.

## Installation

First, activate your conda environment, then you should have your `CONDA_PREFIX` env
variable set.

Then, manually set the `CUDA_HOME` env variable, for example,

```bash
export CUDA_HOME=/mnt/petrelfs/share/cuda-11.8/
```

The CUDA version you set should be compatible with your PyTorch version.

Then, execute `pip install -e . -v` to build the package.

## Usage

See `test/XXX.py` for example usages.

First, import this package:

```python
import torch
import longserve_cuda_kernels
```

**Pay attention that `torch` must be imported before `longserve_cuda_kernels` to avoid dynamic linking error.**

Then you can invoke `longserve_cuda_kernels.XXX` to use the CUDA kernels.
