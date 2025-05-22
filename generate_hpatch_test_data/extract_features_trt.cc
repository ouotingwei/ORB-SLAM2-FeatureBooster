#include <NvInfer.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <bitset>
#include <filesystem>
#include <cstdlib>

#include "cnpy.h"
#include "ORBextractor.h"

using namespace nvinfer1;
using ORB_SLAM2::ORBextractor;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

std::vector<std::string> load_image_list(const std::string& filepath) {
    std::ifstream file(filepath);
    std::vector<std::string> paths;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) paths.push_back(line);
    }
    return paths;
}

void normalize_keypoints(std::vector<cv::KeyPoint>& kps, cv::Mat& out, const cv::Size& img_size) {
    float x0 = img_size.width / 2.0f;
    float y0 = img_size.height / 2.0f;
    float scale = std::max(img_size.width, img_size.height) * 0.7f;

    out = cv::Mat(kps.size(), 4, CV_32F);
    for (size_t i = 0; i < kps.size(); ++i) {
        out.at<float>(i, 0) = (kps[i].pt.x - x0) / scale;
        out.at<float>(i, 1) = (kps[i].pt.y - y0) / scale;
        out.at<float>(i, 2) = kps[i].size / 31.0f;
        out.at<float>(i, 3) = kps[i].angle * CV_PI / 180.0f;
    }
}

ICudaEngine* load_engine(const std::string& engine_path, IRuntime* runtime) {
    std::ifstream file(engine_path, std::ios::binary);
    file.seekg(0, file.end);
    size_t length = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> data(length);
    file.read(data.data(), length);
    return runtime->deserializeCudaEngine(data.data(), length);
}

int main() {
    std::string image_list_file = "/media/wei/T7/hpatches-sequences-release/image_list.txt";
    std::string engine_path = "/home/wei/orb_slam2_booster/src/ORB-SLAM2-FeatureBooster/model/feature_booster_int8.engine";

    cudaSetDevice(0);
    cudaFree(0);
    IRuntime* runtime = createInferRuntime(logger);
    ICudaEngine* engine = load_engine(engine_path, runtime);
    IExecutionContext* context = engine->createExecutionContext();

    ORBextractor orb(3000, 1.2f, 8, 20, 7);
    auto images = load_image_list(image_list_file);

    for (const auto& path : images) {
        cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) continue;

        std::vector<cv::KeyPoint> kps;
        cv::Mat descriptors;
        orb(img, cv::Mat(), kps, descriptors);
        if (kps.empty()) continue;

        cv::Mat desc_float(kps.size(), 256, CV_32F);
        for (int i = 0; i < descriptors.rows; ++i)
            for (int j = 0; j < descriptors.cols; ++j)
                for (int b = 0; b < 8; ++b)
                    desc_float.at<float>(i, j * 8 + b) = ((descriptors.at<uchar>(i, j) >> b) & 1) ? 1.0f : -1.0f;

        cv::Mat kps_norm;
        normalize_keypoints(kps, kps_norm, img.size());

        void *d_desc, *d_kps, *d_out;
        size_t desc_size = desc_float.total() * sizeof(float);
        size_t kps_size = kps_norm.total() * sizeof(float);
        size_t out_size = desc_float.rows * 256 * sizeof(float);

        cudaMalloc(&d_desc, desc_size);
        cudaMalloc(&d_kps, kps_size);
        cudaMalloc(&d_out, out_size);

        cudaMemcpy(d_desc, desc_float.data, desc_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_kps, kps_norm.data, kps_size, cudaMemcpyHostToDevice);

        Dims3 dimDesc{1, desc_float.rows, 256};
        Dims3 dimKps{1, kps_norm.rows, 4};

        context->setInputShape("descriptors", dimDesc);
        context->setInputShape("keypoints", dimKps);
        context->setTensorAddress("descriptors", d_desc);
        context->setTensorAddress("keypoints", d_kps);
        context->setTensorAddress("boosted_descriptors", d_out);

        bool success = context->enqueueV3(nullptr);
        if (!success) {
            std::cerr << "[ERROR] TensorRT inference failed." << std::endl;
            continue;
        }

        cudaDeviceSynchronize();

        std::vector<float> out(desc_float.rows * 256);
        cudaMemcpy(out.data(), d_out, out_size, cudaMemcpyDeviceToHost);

        std::vector<uint8_t> out_bin(desc_float.rows * 32, 0);
        for (int i = 0; i < desc_float.rows; ++i) {
            for (int j = 0; j < 256; ++j) {
                float val = out[i * 256 + j];
                if (val > 0.0f) 
                    out_bin[i * 32 + j / 8] |= (1 << (j % 8));
            }
        }

        std::cout << "[DEBUG] First binary descriptor:" << std::endl;
        for (int j = 0; j < 32; ++j) {
            std::cout << static_cast<int>(out_bin[j]);
            if (j < 31) std::cout << ", ";
        }
        std::cout << std::endl;

        std::vector<float> keypoints;
        float x0 = img.cols / 2.0f;
        float y0 = img.rows / 2.0f;
        float scale = std::max(img.cols, img.rows) * 0.7f;
        for (int i = 0; i < kps.size(); ++i) {
            float x_norm = kps_norm.at<float>(i, 0);
            float y_norm = kps_norm.at<float>(i, 1);
            float size = kps_norm.at<float>(i, 2);
            float angle = kps_norm.at<float>(i, 3);
            float x = x_norm * scale + x0;
            float y = y_norm * scale + y0;
            keypoints.push_back(x);
            keypoints.push_back(y);
            keypoints.push_back(size);
            keypoints.push_back(angle);
        }

        std::string out_dir = path + ".ORB+Boost-B-INT8";
        std::string cmd = "mkdir -p \"" + out_dir + "\"";
        std::system(cmd.c_str());

        std::string orb_kp_name = out_dir + "/keypoints.npy";
        std::string orb_desc_name = out_dir + "/descriptors.npy";

        cnpy::npy_save(orb_kp_name, keypoints.data(), {kps.size(), 4}, "w");
        cnpy::npy_save(orb_desc_name, reinterpret_cast<const unsigned char*>(out_bin.data()), {kps.size(), 32}, "w");

        std::cout << "[INFO] Saved: " << orb_kp_name << std::endl;
        std::cout << "[INFO] Saved: " << orb_desc_name << std::endl;

        cudaFree(d_desc);
        cudaFree(d_kps);
        cudaFree(d_out);
    }

    delete context;
    delete engine;
    delete runtime;
    return 0;
}
