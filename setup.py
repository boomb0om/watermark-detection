from setuptools import setup, find_packages

setup(
    name="wmdetection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "pillow>=8.0.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.50.0",
        "huggingface-hub>=0.4.0",
        "opencv-python>=4.4.0",
        "pandas>=1.1.0",
        "scikit-learn>=0.24.0",
        "timm>=0.6.12",
    ],
    python_requires=">=3.8",
    description="A package for detecting watermarks in images and videos",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 