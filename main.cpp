#include <iostream>
#include <opencv2/opencv.hpp>

#include "yolov4.h"

void init()
{
    if (yolov4::detector == nullptr)
    {
        yolov4::detector = new yolov4(
            "/home/pasan/Projects/HCI/ncnn/cmake_test/models/yolov4-gen-tiny-opt.param",
            "/home/pasan/Projects/HCI/ncnn/cmake_test/models/yolov4-gen-tiny-opt.bin");
    }
}

int main(int, char **)
{
    cv::Mat image;
    cv::namedWindow("Display window");
    cv::VideoCapture cap(0);

    init();

    if (!cap.isOpened())
    {
        std::cout << "cannot open camera" << std::endl;
    }

    while (true)
    {
        cap >> image;

        auto result = yolov4::detector->detect(image, 0.35, 0.7);

        for (auto rec : result)
        {
            cv::rectangle(image, rec.box, cv::Scalar(0, 255, 0), 2, 1);
        }

        cv::imshow("Display window", image);

        // Press ESC to exit
        if ((char)cv::waitKey(25) == 27)
        {
            break;
        }
    }

    return 0;
}
