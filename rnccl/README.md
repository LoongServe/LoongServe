# RNCCL - Raw NCCL Bindings for Python

This library, rnccl, provides a thin wrapper around NCCL C APIs for Python. It is intended to be used as building blocks for communication between SP workers in
LongServe.

## Motivation

Torch's distributed communication package, `torch.distributed`, provides interfaces for
communication. However it is not as flexible as the raw NCCL APIs, and it often
introduces unwanted synchronization, which is not desirable in some cases.

This library, RNCCL, provide low-level APIs for communication, serving as basic building primitives for Computer System programmers to build more flexible communication.

## Installation

Please have `nccl` installed in your conda environment. You can install it via

```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
conda install conda-forge::nccl==2.19.3.1
```

Here we install nccl v2.19.3.1 to align with PyTorch 2.2.0's nccl version to avoid
confusion during `#include`-ing or linking.

Then execute `pip install -e .` to install `rnccl` into your environment.

## Usage

### Setting Up

See `test.py` for an example usage.

First, import this package:

```python
import torch
import rnccl
```

**Pay attention that `torch` must be imported before `rnccl` to avoid dynamic linking error.**

The master process shall create a rnccl_id (analogy to `ncclUniqueId` in C) and pass
it to every worker process.

```python
unique_id = rnccl.get_nccl_unique_id()
```

Every worker process shall create a communicator via

```python
comm = rnccl.RNCCLComm(unique_id, world_size, rank)
```

### Communication

The communicator object, `comm`, provides the following methods:

- `comm.nccl_group_start()`
- `comm.nccl_group_end()`
- `comm.nccl_send()`
- `comm.nccl_recv()`

You can use `nccl_send()` or `nccl_recv()` either within a `nccl_group_start()` and `nccl_group_end()` pair, or without them. The former is useful when you want to
batch multiple `nccl_send()` and `nccl_recv()` calls together to reduce the overhead.

### Synchronization

RNCCL, by default, does not perform any synchronization. You can use the following
two methods:

- `comm.let_default_stream_wait()`: Let the default CUDA stream wait for RNCCL's
  stream, i.e. all future CUDA operations will wait for RNCCL's operations to finish.
- `comm.wait_for_default_stream()`: Wait for the default CUDA stream to finish,
  i.e. all future RNCCL's operations will wait for the default CUDA stream to finish.

The two methods are asynchronous and non-blocking relative to the CPU.
