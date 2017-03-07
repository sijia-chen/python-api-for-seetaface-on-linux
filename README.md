# Python API for SeetaFace on Linux

## Description

This is a python API for SeetaFace on Linux. Based on Seetaface source files, I did some changes in **`CMakeLists`** files and wrote **`so_api.cpp`** in **`src`** folders as cpp interfaces respectively.

## Interfaces

**`face_detect`**:Detects faces in an image.

**`face_align`**:Detects faces in an image, returns 5 landmarks, i.e, 2 left eye centers, nose tip and two mouth corners, in every face.

**`face_verify`**:Detects only **one** face and its landmarks in every image, calculates the similarity among these two faces.

see **`seetaface_api.py`** for more details.

## Required Libs

- openCV
- cmake

## How to Build Share Libs

Just follow the steps in README.md in every folders.

## Tips

If you have any trouble with **`-lopencv_dep_cudart`**, try **`cmake .. -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF`** from the very beginning.

## Sample

![Aaron_Peirsol_0001](https://ooo.0o0.ooo/2017/03/07/58be302547a7b.jpg)

## About Seetaface

**SeetaFace Engine** is an open source C++ face recognition engine, which can run on CPU with no third-party dependence. It contains three key parts, i.e., **SeetaFace Detection**, **SeetaFace Alignment** and **SeetaFace Identification**, which are necessary and sufficient for building a real-world face recognition applicaiton system.

see [this link](https://github.com/seetaface/SeetaFaceEngine) for more details.
