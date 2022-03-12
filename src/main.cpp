#include <iostream>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include "yolov4.h"
#include "tracker.h"

const std::string VIDEO_PATH = "../inputs/";
const std::string MODEL_PATH = "../models/";

cv::VideoCapture cap;
int g_slider_position = 0;
int g_run = 1, g_dontset = 0;

void modelInit()
{
    if (yolov4::detector == nullptr)
    {
        yolov4::detector = new yolov4(
            (MODEL_PATH + "custom-yolov4-tiny-detector_opt.param").c_str(),
            (MODEL_PATH + "custom-yolov4-tiny-detector_opt.bin").c_str());
    }
}

void onTrackbarSlide(int pos, void *)
{
    cap.set(cv::CAP_PROP_POS_FRAMES, pos);
    if (!g_dontset)
        g_run = 1;
    g_dontset = 0;
}

int main(int argc, char *argv[], char *envp[])
{
    std::string src = VIDEO_PATH;
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " VIDEO SOURCE" << std::endl;
        return 1;
    }
    src = src + argv[1];
    std::cout << "INPUT VIDEO : " << src << std::endl;

    cv::Mat image;
    cv::namedWindow("test", cv::WINDOW_AUTOSIZE);
    Tracker tracker;

    cap.open(src.c_str());

    modelInit();

    if (!cap.isOpened())
        std::cout << "cannot open camera" << std::endl;

    int frames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    cv::createTrackbar("Position", "test", &g_slider_position, frames, onTrackbarSlide);

    while (cap.isOpened())
    {
        cap >> image;
        int current_pos = (int)cap.get(cv::CAP_PROP_POS_FRAMES);
        cv::setTrackbarPos("Position", "test", current_pos);

        auto result = yolov4::detector->detect(image, 0.35, 0.7);

        tracker.Run(result);
        const auto tracks = tracker.GetTracks();

        // for (auto rec : result)
        // {
        //     cv::rectangle(image, rec.box, cv::Scalar(0, 255, 0), 2, 1);
        //     std::string label = yolov4::detector->labels.at(rec.label) + " " + std::to_string(rec.score) + " %";
        //     cv::putText(image,
        //                 label,
        //                 cv::Point(rec.box.x, rec.box.y),
        //                 cv::FONT_HERSHEY_DUPLEX,
        //                 1.0,
        //                 CV_RGB(0, 255, 0),
        //                 1);
        // }

        for (auto &trk : tracks)
        {
            // only draw tracks which meet certain criteria
            if (trk.second.coast_cycles_ < kMaxCoastCycles &&
                (trk.second.hit_streak_ >= kMinHits))
            {
                const auto &bbox = trk.second.GetStateAsBbox();
                std::string label = std::to_string(trk.first) + " : " + yolov4::detector->labels.at(trk.second.label);
                cv::putText(image,
                            label,
                            cv::Point(bbox.tl().x, bbox.tl().y - 10),
                            cv::FONT_HERSHEY_DUPLEX,
                            1.0,
                            CV_RGB(0, 255, 0),
                            1);
                cv::rectangle(image, bbox, cv::Scalar(0, 255, 0), 2, 1);
            }
        }

        cv::imshow("test", image);
        char key = (char)cv::waitKey(10);
        // Press ESC to exit
        if (key == 27)
        {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
