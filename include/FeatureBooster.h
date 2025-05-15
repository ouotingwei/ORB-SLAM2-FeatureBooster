#include "onnxruntime_cxx_api.h"
#include <opencv2/core.hpp>
#include <bitset>
#include <mutex>
#include <iostream>

namespace ORB_SLAM2
{
    
class FeatureBooster
{
public:
    FeatureBooster(int img_w, int img_h);
    bool LoadOnnxModel(const std::string &model_path);
    void BoostDescriptors(std::vector<cv::KeyPoint> &kps, cv::Mat &Desc);

private:
    int img_w, img_h;

    std::mutex mutex_inference;

    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;
};

}