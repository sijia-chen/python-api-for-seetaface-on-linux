# Python API for SeetaFace on Linux

This is a python API for SeetaFace on Linux. Based on Seetaface source files, I did some changes in **CMakeLists** files and wrote **so_api.cpp** in **src** folders as cpp interfaces respectively.

## Required Libs

openCV
cmake

## How to Build Share Libs

Just follow the steps in README.md in every folders.

## Tips

If you have any trouble with **-lopencv_dep_cudart**, try **cmake .. -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF** from the very beginning.
