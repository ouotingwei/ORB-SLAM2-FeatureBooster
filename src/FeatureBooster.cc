// FeatureBooster.cpp（已加入 DEBUG 訊息）
#include "FeatureBooster.h"
#include <NvInfer.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>

namespace ORB_SLAM2 {

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

FeatureBooster::FeatureBooster(int w, int h, const std::string& engine_path)
    : img_w(w), img_h(h)
{
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        std::cerr << "[FeatureBooster] cudaSetDevice failed: " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    cudaFree(0);

    std::cerr << "[DEBUG] Constructor: runtime creating..." << std::endl;
    runtime = nvinfer1::createInferRuntime(logger);
    std::cerr << "[DEBUG] Constructor: calling LoadEngine()" << std::endl;
    bool ok = LoadEngine(engine_path);
    std::cerr << "[DEBUG] Constructor: LoadEngine returned " << ok << std::endl;
    cudaStreamCreate(&stream);
}

FeatureBooster::~FeatureBooster()
{
    std::cerr << "[DEBUG] Destructor: destroying stream and freeing resources" << std::endl;
    cudaStreamDestroy(stream);
    delete context;
    delete engine;
    delete runtime;
}

bool FeatureBooster::LoadEngine(const std::string& engine_path)
{
    std::cerr << "[FeatureBooster] LoadEngine start, path=" << engine_path << std::endl;
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "[FeatureBooster] Failed to open engine file" << std::endl;
        return false;
    }
    size_t size = file.tellg();
    std::cerr << "[FeatureBooster] Engine file size = " << size << " bytes" << std::endl;
    file.seekg(0, std::ios::beg);

    std::vector<char> buf(size);
    file.read(buf.data(), size);
    if (!file) {
        std::cerr << "[FeatureBooster] Failed to read engine file into buffer" << std::endl;
        return false;
    }

    std::cerr << "[FeatureBooster] Calling deserializeCudaEngine()" << std::endl;
    engine = runtime->deserializeCudaEngine(buf.data(), size);
    if (!engine) {
        std::cerr << "[FeatureBooster] deserializeCudaEngine returned nullptr" << std::endl;
        return false;
    }
    std::cerr << "[FeatureBooster] Engine deserialized successfully" << std::endl;

    std::cerr << "[FeatureBooster] Creating ExecutionContext" << std::endl;
    context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "[FeatureBooster] createExecutionContext returned nullptr" << std::endl;
        return false;
    }
    std::cerr << "[FeatureBooster] ExecutionContext created successfully" << std::endl;
    return true;
}

void FeatureBooster::BoostDescriptors(std::vector<cv::KeyPoint> &kps, cv::Mat &Desc) {
    std::lock_guard<std::mutex> l(mutex_inference);

    int N = kps.size();
    const int D = 256;
    if (N == 0 || Desc.empty()) return;

    // std::cerr << "[DEBUG] BoostDescriptors called, num_kps=" << N
    //           << ", Desc.rows=" << Desc.rows << ", Desc.cols=" << Desc.cols << std::endl;

    std::vector<float> kp_in;    kp_in.reserve(N * 4);
    std::vector<float> desc_in;  desc_in.reserve(N * D);

    float x0 = img_w / 2.f;
    float y0 = img_h / 2.f;
    float scale = std::max(img_w, img_h) * 0.7f;

    for (auto &kp : kps) {
        kp_in.push_back((kp.pt.x - x0) / scale);
        kp_in.push_back((kp.pt.y - y0) / scale);
        kp_in.push_back(kp.size / 31.f);
        kp_in.push_back(kp.angle * CV_PI / 180.f);
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < 32; j++) {
            unsigned char b = Desc.at<unsigned char>(i, j);
            for (int t = 0; t < 8; t++)
                desc_in.push_back((b >> t) & 1 ? 1.f : -1.f);
        }
    }

    size_t S_kp = N * 4 * sizeof(float);
    size_t S_dc = N * D * sizeof(float);

    cudaMalloc(&kp_dev,   S_kp);
    cudaMalloc(&desc_dev, S_dc);
    cudaMalloc(&out_dev,  S_dc);

    cudaMemcpyAsync(kp_dev,   kp_in.data(),   S_kp, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(desc_dev, desc_in.data(), S_dc, cudaMemcpyHostToDevice, stream);

    nvinfer1::Dims3 dimKp{1, N, 4};
    nvinfer1::Dims3 dimDesc{1, N, D};

    context->setInputShape("keypoints",   dimKp);
    context->setInputShape("descriptors", dimDesc);

    context->setTensorAddress("keypoints",   kp_dev);
    context->setTensorAddress("descriptors", desc_dev);
    context->setTensorAddress("boosted_descriptors", out_dev);

    auto t0 = std::chrono::high_resolution_clock::now();
    context->enqueueV3(stream);
    cudaStreamSynchronize(stream);
    auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<float> out(N * D);
    cudaMemcpyAsync(out.data(), out_dev, S_dc, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < 32; j++) {
            unsigned char v = 0;
            for (int t = 0; t < 8; t++)
                if (out[i * D + j * 8 + t] >= 0.0f)
                    v |= (1 << t);
            Desc.at<unsigned char>(i, j) = v;
        }
    }

    cudaFree(kp_dev);
    cudaFree(desc_dev);
    cudaFree(out_dev);

    double dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << std::fixed << std::setprecision(2) << dt << " ms, " << N << " kps." << std::endl;
}


} // namespace ORB_SLAM2