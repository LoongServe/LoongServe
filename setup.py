from setuptools import setup, find_packages

setup(
    name="loongserve",
    version="2.0.0",
    packages=find_packages(
        exclude=("build", "include", "test", "dist", "docs", "benchmarks", "loongserve.egg-info")
    ),
    author="model toolchain",
    author_email="",
    description="elastic distributed LLM serving system",
    long_description="",
    long_description_content_type="text/markdown",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Linux",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pyzmq",
        "uvloop",
        # "torch",
        "transformers",
        "einops",
        "packaging",
        "rpyc",
        "ninja",
        "safetensors",
        # "triton",
        "fastapi",
        "pillow",
        "ray",
        "sentencepiece",
        "aiohttp"
    ],
)
