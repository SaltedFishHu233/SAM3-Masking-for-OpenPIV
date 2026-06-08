# SAM3-Masking-for-OpenPIV
## Introduction
This library stores the integrated workflow of SAM3 and OpenPIV-python-cpu. SAM3 is used to automatically generate mask for the PIV analysis conducted through OpenPIV-python-cpu
## Installation
To install and run the repository, make sure CUDA is installed and configured correctly.

Upon doing so, in the desired environment, setup SAM3 according to installation instruction discussed in the SAM3 repo: https://github.com/facebookresearch/sam3


after doing so, run

```console
pip install git+https://github.com/ali-sh-96/openpiv-python-cpu
```

to install oepnpiv-python-cpu

## Notebook Description
There are three notebooks for this repository:

SAM3MaskGen: Step by step workflow of the automated script in the Scripts folder

AutoSAM3: Simplified mask generation to test the script in Scripts folder

OpenPIVwSAM3: Integrated workflow of PIV analysis with the automated mask generation using SAM3
