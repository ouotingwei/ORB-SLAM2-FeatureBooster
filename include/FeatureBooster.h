#ifndef FEATUREBOOSTER_H
#define FEATUREBOOSTER_H

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <mutex>
#include <string>

namespace ORB_SLAM2 {

class FeatureBooster {
public:
    FeatureBooster(int w, int h, const std::string& engine_path);
    ~FeatureBooster();
    bool LoadEngine(const std::string& engine_path);
    void BoostDescriptors(std::vector<cv::KeyPoint>& kps, cv::Mat& Desc);

private:
    int img_w, img_h;
    nvinfer1::IRuntime* runtime{nullptr};
    nvinfer1::ICudaEngine* engine{nullptr};
    nvinfer1::IExecutionContext* context{nullptr};
    cudaStream_t stream{0};
    void* desc_dev{nullptr};
    void* kp_dev{nullptr};
    void* out_dev{nullptr};
    std::mutex mutex_inference;
    static constexpr const char* DESC_NAME = "descriptors";
    static constexpr const char* KP_NAME   = "keypoints";
    static constexpr const char* OUT_NAME  = "boosted_descriptors";
};

}

#endif