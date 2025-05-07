// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "image_enc.h"

// Expand the image into a square and fill it with the specified background color
cv::Mat expand2square(const cv::Mat& img, const cv::Scalar& background_color) {
    int width = img.cols;
    int height = img.rows;

    // If the width and height are equal, return to the original image directly
    if (width == height) {
        return img.clone();
    }

    // Calculate the new size and create a new image
    int size = std::max(width, height);
    cv::Mat result(size, size, img.type(), background_color);

    // Calculate the image paste position
    int x_offset = (size - width) / 2;
    int y_offset = (size - height) / 2;

    // Paste the original image into the center of the new image
    cv::Rect roi(x_offset, y_offset, width, height);
    img.copyTo(result(roi));

    return result;
}

class ImageEncoderService {
private:
    rknn_app_context_t rknn_app_ctx;
    bool model_loaded;

public:
    ImageEncoderService() : model_loaded(false) {
        memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    }

    bool loadModel(const std::string& model_path) {
        if (model_loaded) return true;

        int ret = init_imgenc(model_path.c_str(), &rknn_app_ctx);
        if (ret != 0) {
            std::cout << "init_imgenc failed! ret=" << ret << std::endl;
            return false;
        }
        model_loaded = true;
        return true;
    }

    bool encodeImage(const std::string& image_path, float* output_vec) {
        if (!model_loaded) {
            std::cout << "Model not loaded" << std::endl;
            return false;
        }

        cv::Mat img = cv::imread(image_path);
        if (img.empty()) {
            std::cout << "Failed to load image: " << image_path << std::endl;
            return false;
        }

        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        cv::Scalar background_color(127.5, 127.5, 127.5);
        cv::Mat square_img = expand2square(img, background_color);
        cv::Mat resized_img;
        cv::Size new_size(392, 392);
        cv::resize(square_img, resized_img, new_size, 0, 0, cv::INTER_LINEAR);

        int ret = run_imgenc(&rknn_app_ctx, resized_img.data, output_vec);
        if (ret != 0) {
            std::cout << "Encoding failed" << std::endl;
            return false;
        }

        return true;
    }

    ~ImageEncoderService() {
        if (model_loaded) {
            release_imgenc(&rknn_app_ctx);
        }
    }
};

static ImageEncoderService encoder_service;

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " model_path\n";
        return -1;
    }

    const char * model_path = argv[1];

    if (!encoder_service.loadModel(model_path)) {
        std::cout << "Failed to load model" << std::endl;
        return -1;
    }

    std::string line;
    float img_vec[196 * 1536];
    
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;

        size_t separator = line.find('|');
        if (separator == std::string::npos) {
            std::cout << "Invalid input format, expected 'image_path|output_path'" << std::endl;
            continue;
        }

        std::string image_path = line.substr(0, separator);
        std::string output_path = line.substr(separator + 1);

        std::chrono::high_resolution_clock::time_point t_start_us = std::chrono::high_resolution_clock::now();

        if (encoder_service.encodeImage(image_path, img_vec)) {
            // Write binary vector to file
            std::ofstream out_file(output_path, std::ios::binary);
            if (!out_file) {
                std::cout << "Error: Failed to open output file: " << output_path << std::endl;
                continue;
            }
            out_file.write(reinterpret_cast<char*>(img_vec), sizeof(img_vec));
            out_file.close();

            std::chrono::high_resolution_clock::time_point t_end_us = std::chrono::high_resolution_clock::now();
            auto encoder_time = std::chrono::duration_cast<std::chrono::microseconds>(t_end_us - t_start_us);

            std::cout << "Success: encoder the image cost " << encoder_time.count() / 1000.0 << "ms" << std::endl;
        } else {
            std::cout << "Error: Failed to encode image: " << image_path << std::endl;
        }
    }
    return 0;
}
