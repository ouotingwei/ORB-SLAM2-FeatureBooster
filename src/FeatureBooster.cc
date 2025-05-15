#include "FeatureBooster.h"
#include <chrono>
#include <iomanip>

namespace ORB_SLAM2 {

FeatureBooster::FeatureBooster(int w, int h)
    : img_w(w), img_h(h), env(ORT_LOGGING_LEVEL_WARNING, "FeatureBooster")
{
    OrtCUDAProviderOptions cuda_options{};
    session_options.AppendExecutionProvider_CUDA(cuda_options);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
}

bool FeatureBooster::LoadOnnxModel(const std::string &model_path)
{
    try {
        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
        std::cout << "Successfully loaded ONNX model: " << model_path << std::endl;
        return true;
    } catch (const Ort::Exception &e) {
        std::cout << "Failed to load ONNX model: " << model_path << "\nError: " << e.what() << std::endl;
        return false;
    }
}

void FeatureBooster::BoostDescriptors(std::vector<cv::KeyPoint> &kps, cv::Mat &Desc)
{
    std::lock_guard<std::mutex> lock(mutex_inference);

    auto start = std::chrono::high_resolution_clock::now();

    int num_kps = kps.size();
    int desc_dim = 256;

    if (num_kps == 0 || Desc.empty())
        return;

    // keypoint preprocess
    std::vector<float> keypoints_input;
    keypoints_input.reserve(num_kps * 4);
    float x0 = img_w / 2.0f, y0 = img_h / 2.0f;
    float scale = std::max(img_h, img_w) * 0.7f;

    for (auto &kp : kps) {
        float x = (kp.pt.x - x0) / scale;
        float y = (kp.pt.y - y0) / scale;
        float s = kp.size / 31.f;
        float a = kp.angle * CV_PI / 180.0f;
        keypoints_input.insert(keypoints_input.end(), {x, y, s, a});
    }

    // descriptor preprocess
    std::vector<float> descriptors_input;
    descriptors_input.reserve(num_kps * desc_dim);
    for (int i = 0; i < num_kps; ++i) {
        for (int j = 0; j < 32; ++j) {
            uchar byte = Desc.at<uchar>(i, j);
            std::bitset<8> bits(byte);
            for (int b = 0; b < 8; ++b) {
                descriptors_input.push_back(bits[b] ? 1.0f : -1.0f);
            }
        }
    }

    // inference
    std::array<int64_t, 2> kp_shape{num_kps, 4};
    std::array<int64_t, 2> desc_shape{num_kps, desc_dim};

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value kp_tensor = Ort::Value::CreateTensor<float>(memory_info, keypoints_input.data(), keypoints_input.size(), kp_shape.data(), 2);
    Ort::Value desc_tensor = Ort::Value::CreateTensor<float>(memory_info, descriptors_input.data(), descriptors_input.size(), desc_shape.data(), 2);

    std::vector<const char*> input_names = {"descriptors", "keypoints"};
    std::vector<const char*> output_names = {"boosted_descriptors"};
    std::array<Ort::Value, 2> input_tensors = {std::move(desc_tensor), std::move(kp_tensor)};

    auto output_tensors = session->Run(Ort::RunOptions{nullptr},
                                    input_names.data(),
                                    input_tensors.data(),
                                    input_tensors.size(),
                                    output_names.data(),
                                    1);

    float* output_ptr = output_tensors.front().GetTensorMutableData<float>();

    // overwrite Desc
    for (int i = 0; i < num_kps; ++i) {
        for (int j = 0; j < 32; ++j) {
            uchar byte = 0;
            for (int b = 0; b < 8; ++b) {
                if (output_ptr[i * 256 + (j * 8 + b)] >= 0.0f)
                    byte |= (1 << b);
            }
            Desc.at<uchar>(i, j) = byte;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> inference_time = end - start;
    std::cout << "Inference Time: " << std::fixed << std::setprecision(2)
              << inference_time.count() << " ms, " << num_kps << " kps."<< std::endl;
}

}