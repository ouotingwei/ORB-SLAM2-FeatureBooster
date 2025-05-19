#include "FeatureBooster.h"
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/stat.h>

using namespace ORB_SLAM2;

size_t GetFileSize(const std::string& path) {
    struct stat stat_buf;
    return stat(path.c_str(), &stat_buf) == 0 ? stat_buf.st_size : 0;
}

int main() {
    std::string engine_path = "/home/wei/orb_slam2_booster/src/ORB-SLAM2-FeatureBooster/model/feature_booster_int8.engine";
    int img_w = 640, img_h = 480;

    FeatureBooster booster(img_w, img_h, engine_path);

    std::vector<int> num_points_list = {500, 1000, 2000, 4000, 8000};

    // warm-up
    {
        std::vector<cv::KeyPoint> kps;
        cv::Mat desc(1000, 32, CV_8U);

        for (int i = 0; i < 1000; ++i) {
            float x = rand() % img_w;
            float y = rand() % img_h;
            float size = 10 + rand() % 10;
            float angle = rand() % 360;
            kps.emplace_back(cv::Point2f(x, y), size, angle);
            for (int j = 0; j < 32; ++j)
                desc.at<uchar>(i, j) = rand() % 256;
        }

        booster.BoostDescriptors(kps, desc);
    }

    for (int num_points : num_points_list) {
        std::cout << "--- Testing " << num_points << " keypoints ---" << std::endl;

        std::vector<cv::KeyPoint> kps;
        cv::Mat desc(num_points, 32, CV_8U);

        for (int i = 0; i < num_points; ++i) {
            float x = rand() % img_w;
            float y = rand() % img_h;
            float size = 10 + rand() % 10;
            float angle = rand() % 360;
            kps.emplace_back(cv::Point2f(x, y), size, angle);
            for (int j = 0; j < 32; ++j)
                desc.at<uchar>(i, j) = rand() % 256;
        }

        booster.BoostDescriptors(kps, desc);
    }

    std::cout << "--- Model Info ---" << std::endl;
    std::cout << "Model Path: " << engine_path << std::endl;
    std::cout << "Model Size: " << GetFileSize(engine_path) / 1024.0 << " KB" << std::endl;

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "GPU Memory Usage: " 
              << (total_mem - free_mem) / 1024.0 / 1024.0 
              << " MB used / " << total_mem / 1024.0 / 1024.0 << " MB total" << std::endl;

    return 0;
}
