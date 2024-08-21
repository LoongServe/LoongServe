# Environment Setup Guide

In this document, we provide instructions on how to set up the environment for running the experiments in the paper.

It takes a relatively long time (about 2 human hours and 4 machine hours) to set up all the environments. To save your time, we have prepared the environments for you. If you want to reproduce the environment, please follow the instructions below.

**Every command below with a `$` prefix indicates that this command needs to be run on a compute node, i.e. you should use `srun` to run it**

## Setting up Environment for LoongServe

Clone the code from GitHub:

```bash
cd ~/research/
git clone https://github.com/BingyangWu/LongServe.git
cd LongServe
```

Initialize the conda environment:

```bash
conda create -n loongserve python=3.12
```

Install the required packages:

```bash
$ conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install pybind11
pip install aiohttp transformers fastapi ninja pyzmq rpyc sentencepiece uvloop einops ray[default] uvicorn
pip install matplotlib ipykernel pandas
```

Install LoongServe:

```bash
cd longserve_c_scheduler/
$ pip install -e .
cd ..

cd longserve_cuda_kernels/
$ pip install -e .
cd ..

cd rnccl
$ pip install -e .
cd ..

$ pip install -e .
```

## Setting up Environment for vLLM

First, clone the code from GitHub, and switch to the correct branch via the following command. This branch is based on vLLM v0.3.0 (the latest version when we conducted the experiments) with some minor modifications to support the LoongServe system. You may refer to the [commit history](https://github.com/interestingLSY/vllm/commits/loongserve-baseline-vllm/) for the exact changes.

```bash
cd ~/research/
git clone https://github.com/interestingLSY/vllm.git
cd vllm
git checkout loongserve-baseline-vllm
```

Create the conda environment:

```bash
conda env create -n vllm -f shanghai-ai-lab-environment.yml
```

Install vLLM:

```bash
conda activate vllm
$ pip install -e .
```

## Setting up Environment for LightLLM w/ SplitFuse

First, clone the code from GitHub, and switch to the correct branch via commands below. This branch is based on the latest version of LightLLM when we conducted the experiments. 

```bash
cd ~/research/
git clone https://github.com/interestingLSY/lightllm.git
cd lightllm
git checkout loongserve-baseline
```

Create the conda environment:

```bash
export PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu118
conda env create -n lightllm-sf -f shanghai-ai-lab-environment.yml
```

Install LightLLM:

```bash
conda activate lightllm-sf
pip install triton==2.1.0	# Upgrade triton to 2.1.0 as required by LightLLM
pip install -e . --no-deps
```

## Setting up Environment for DeepSpeed MII (Dynamic SplitFuse)

First, clone the code from GitHub, and switch to the correct branch via commands below. This branch is based on DeepSpeed-MII v0.2.3 (the latest version when we conducted the experiments) with some bug fixes. 

```bash
cd ~/research
git clone https://github.com/interestingLSY/DeepSpeed-MII.git
cd DeepSpeed-MII
git checkout loongserve-baseline-deepspeed
```

Create the conda environment:

```bash
conda env create -n deepspeed -f shanghai-ai-lab-environment.yml
```

Install DeepSpeed-MII:

```bash
conda activate deepspeed
$ pip install -e .
```

## Setting up Environment for DistServe (Prefill-Decoding Disaggregation)

First, clone the code from GitHub. We use the latest branch of DistServe.

```bash
cd ~/research
git clone https://github.com/LLMServe/DistServe.git
cd DistServe
```

Create the conda environment:

```bash
conda env create -n distserve -f environment.yml
```

Install SwiftTransformer, a dependency of DistServe:

```bash
conda activate distserve
git clone https://github.com/LLMServe/SwiftTransformer.git
cd SwiftTransformer
git submodule update --init --recursive

export NVCC_PREPEND_FLAGS="-ccbin /mnt/petrelfs/share/gcc/gcc-11.2.0/bin/g++"
conda install conda-forge::cmake	# The CMake on Shanghai AI Lab is too old
conda install conda-forge::openmpi
$ conda install conda-forge::nccl
$ cmake -B build .
$ cmake --build build -j64
cd ..
```

Install DistServe:

```bash
pip install -e .
```
