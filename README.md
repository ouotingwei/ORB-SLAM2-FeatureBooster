# ORB-SLAM2 + FeatureBooster

This project is an example of integrating **ORB-SLAM2** with **FeatureBooster** (CVPR 2023). In addition to the model provided by FeatureBooster, we have also performed model quantization to further reduce the memory usage and inference time, making it more efficient on edge GPU for real-time applications.

## Performance
- Inference Time

- EuRoC MAV Dataset


## Requirements

- Ubuntu 20.04
- OpenCV 4.2.0
- Eigen 3.3.9
- Pangolin 0.6
- TensorRT 10

[!] This repo was tested on RTX2050 mobile (CUDA 12.1 + CUDNN 8.9.2)

## Installation
1. clone the repo
  ```
  git clone https://github.com/ouotingwei/ORB-SLAM2-FeatureBooster.git
  cd ORB-SLAM2-FeatureBooster
  ```

2. generate .engine (You must recompile this on the device where it will be executed !)

  ```
  cd model
  ```

    - .onnx -> .engine (fp32)

    ```
    trtexec \
    --onnx=feature_booster_explicit.onnx \
    --saveEngine=feature_booster_fp32.engine \
    --memPoolSize=workspace:4096 \
    --profile=0 \
    --minShapes=descriptors:1x1x256,keypoints:1x1x4 \
    --optShapes=descriptors:1x1000x256,keypoints:1x1000x4 \
    --maxShapes=descriptors:1x2200x256,keypoints:1x2200x4
    ```

    - .onnx -> .engine (fp16)

    ```
    trtexec \
    --onnx=feature_booster_explicit.onnx \
    --saveEngine=feature_booster_fp16.engine \
    --fp16 \
    --memPoolSize=workspace:4096 \
    --profile=0 \
    --minShapes=descriptors:1x1x256,keypoints:1x1x4 \
    --optShapes=descriptors:1x1000x256,keypoints:1x1000x4 \
    --maxShapes=descriptors:1x2200x256,keypoints:1x2200x4
    ```

    - .onnx -> .engine (int8)
    ```
    ```

3. modify model path in system.cc (Line:70)
```
std::string enginePath = "path_to_your.engine"; 
```

4. build
```
./build.sh
```

## Run
```
./Examples/Monocular/mono_euroc Vocabulary/ORBvoc.txt Examples/Monocular/EuRoC.yaml PATH_TO_SEQUENCE_FOLDER/mav0/cam0/data Examples/Monocular/EuRoC_TimeStamps/SEQUENCE.txt 
```