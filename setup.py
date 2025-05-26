# setup.py
from setuptools import setup, find_packages

setup(
    name="netvlad_tensorrt",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy==1.23.2",
        "onnx==1.18.0",
        "onnxruntime==1.12.0",
        "onnxscript==0.2.6",
        "protobuf==6.31.0",
        "torch==1.12.0",
        "torchvision==0.13.0",
    ],
)