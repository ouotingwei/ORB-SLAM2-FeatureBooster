# ORB-SLAM2 + FeatureBooster

This project is an example of integrating **[ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2.git)** with **[FeatureBooster](https://github.com/SJTU-ViSYS/FeatureBooster)** (CVPR 2023). In addition to the model provided by FeatureBooster, we have also performed model quantization to further reduce the memory usage and inference time, making it more efficient on edge GPU for real-time applications.

## Performance

- Inference Time (ms)

|                          | #500 | #1000 | #2000 | #4000 | #8000 |
|--------------------------|------|-------|-------|-----------|-----------|
| FeatureBooster (from paper)<br>RTX 3090 | 1.6  | 2.0   | 3.2   | **4.3**   | **7.8**   |
| TensorRT-FP32<br>RTX 2050 mobile       | 1.75 | 2.29  | 4.37  | 9.07      | 20.03     |
| TensorRT-FP16<br>RTX 2050 mobile       | **1.0** | **1.46** | **2.54** | 4.80      | 9.65      |
| TensorRT-INT8<br>RTX 2050 mobile       | 0.91 | 1.83 | 3.64 | 7.09      | 14.29      |

- Model Size

| Platform         | Model Size (KB) |
|------------------|-----------------|
| TensorRT-FP32 (RTX 2050 mobile) | 11476.04 |
| TensorRT-FP16 (RTX 2050 mobile) | 6474.66  |
| TensorRT-INT8 (RTX 2050 mobile) | 4975.55  |


- EuRoC MAV Dataset

1. ORB v.s. ORB+FB(FP32)

|        | MH01     | MH02     | MH03     | MH04     | MH05     | V101     | V102     | V103     | V201     | V202     | V203     |
|--------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| ORB    | **0.044016** | 0.038369 | 0.040270 | **0.058489** | **0.050257** | 0.095751 | 0.066090 | 0.232308 | **0.054922** | **0.057037** | 0.141280 |
| ORB+FB(FP32)   | 0.044181 | **0.034755** | **0.037757** | 0.059651 | 0.052331 | **0.095685** | **0.062482** | **0.064288** | 0.059244 | 0.057042 | **0.069158** |

2. ORB v.s. ORB+FB(FP16)

|        | MH01     | MH02     | MH03     | MH04     | MH05     | V101     | V102     | V103     | V201     | V202     | V203     |
|--------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| ORB    | 0.044016 | 0.038369 | **0.040270** | 0.058489 | 0.050257 | **0.095751** | 0.066090 | 0.232308 | **0.054922** | 0.057037 | 0.141280 |
| ORB+FB(FP16)   | **0.043651** | **0.035909** | 0.041634 | **0.054122** | **0.047836** | 0.096148 | **0.063587** | **0.067077** | 0.058811 | **0.056661** | **0.070641** |

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
--optShapes=descriptors:1x4000x256,keypoints:1x4000x4 \
--maxShapes=descriptors:1x8000x256,keypoints:1x8000x4
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
--optShapes=descriptors:1x4000x256,keypoints:1x4000x4 \
--maxShapes=descriptors:1x8000x256,keypoints:1x8000x4
```

- .onnx -> .engine (int8)
```
...
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

## Speed test
```
cd inference_speed_test
cmake ..
make -j
./main_exec
```