#ifndef YOLOV4_H
#define YOLOV4_H

#include "ncnn/net.h"
#include <opencv2/core/utility.hpp>

namespace vc
{
    typedef struct
    {
        int width;
        int height;
    } Size;
}

typedef struct
{
    std::string name;
    int stride;
    std::vector<vc::Size> anchors;
} YoloLayerData;

typedef struct BoxInfo
{
    cv::Rect box;
    float score;
    int label;

} BoxInfo;

class yolov4
{
public:
    yolov4(const char *param, const char *bin);

    ~yolov4();

    std::vector<BoxInfo> detect(const cv::Mat &_frame, float threshold, float nms_threshold);

    std::vector<std::string> labels{
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"};

private:
    static std::vector<BoxInfo> decode_infer(ncnn::Mat &data, const vc::Size &frame_size, float threshold);

    ncnn::Net *Net;
    int input_size = 416;

    static void nms(std::vector<BoxInfo> &input_boxes, float NMS_THRESH);

public:
    static yolov4 *detector;
};

#endif
