# VIDEO SYSTEM
**FAST,EASY-LEARNING AI VIDEO PROJECT** <br>

- This project, built upon PyAV (for video streaming) and TensorRT (for YOLO model inference), achieves real-time analysis and display of 8 video streams simultaneously. It encompasses the following core functionalities:

  - API-style invocation interface

  - Video stream acquisition

  - Model inference execution

  The implementation leverages TensorRT to optimize inference speed, ensuring high-performance processing across all concurrent streams.
# INTRODUCTION
- **please install cuda,cudnn and tensorrt before learning this project<br>**
- here is the installation of [TensorRT](https://blog.csdn.net/soulproficiency/article/details/150934136?spm=1001.2014.3001.5502)

## USAGE
### 1. model export
- running the below code after installing Tensorrt 
  ```bash
    trtexec --onnx=model.onnx \
    --fp16 \
    --saveEngine=model_int8.engine \
    --workspace=4096
  ```
- move the exported engine file to /video_system/weights

### 2. environment installation

- recommend conda environment
    ```bash
    pip install -r requirement.txt
    ```
### 3. run main.py

### 4. run port_test.py