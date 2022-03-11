#include "yolov4.h"

yolov4 *yolov4::detector = nullptr;

yolov4::yolov4(const char *param, const char *bin)
{
    Net = new ncnn::Net();

    // Net->opt.use_fp16_arithmetic = true; // FP16 arithmetic acceleration
    Net->load_param(param);
    Net->load_model(bin);
}

yolov4::~yolov4()
{
    Net->clear();
    delete Net;
}

std::vector<BoxInfo>
yolov4::detect(const cv::Mat &_frame, float threshold, float nms_threshold)
{
    int img_h = _frame.size().height;
    int img_w = _frame.size().width;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(_frame.data, ncnn::Mat::PIXEL_BGR, _frame.cols, _frame.rows, input_size, input_size);
    float mean[3] = {0, 0, 0};
    float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};

    in.substract_mean_normalize(mean, norm);

    ncnn::Mat out;

    ncnn::Extractor ex = Net->create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);
    ex.input(0, in);
    ex.extract("output", out);

    std::vector<BoxInfo> result;

    auto boxes = decode_infer(out, {(int)img_w, (int)img_h}, threshold);
    result.insert(result.begin(), boxes.begin(), boxes.end());
    // nms(result, nms_threshold);
    return result;
}
/**
 * Calculate the actual parameter value of bbox from the Mat->data value
 * @param data
 * @param frame_size
 * @param threshold
 * @return
 */
std::vector<BoxInfo>
yolov4::decode_infer(ncnn::Mat &data, const vc::Size &frame_size, float threshold)
{
    std::vector<BoxInfo> result;
    for (int i = 0; i < data.h; i++)
    {
        BoxInfo box;
        const float *values = data.row(i);
        box.box.x = values[2] * (float)frame_size.width;
        box.box.y = values[3] * (float)frame_size.height;
        box.box.width = values[4] * (float)frame_size.width - box.box.x;   // x2
        box.box.height = values[5] * (float)frame_size.height - box.box.y; // y2
        box.label = (int)values[0] - 1;
        box.score = values[1];
        if (box.score < threshold)
        {
            continue;
        }
        result.push_back(box);
    }
    return result;
}

void yolov4::nms(std::vector<BoxInfo> &input_boxes, float NMS_THRESH)
{
    std::sort(input_boxes.begin(), input_boxes.end(),
              [](BoxInfo a, BoxInfo b)
              { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        vArea[i] = input_boxes.at(i).box.area();
    }
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        for (int j = i + 1; j < int(input_boxes.size());)
        {
            float xx1 = std::max(input_boxes[i].box.x, input_boxes[j].box.x);
            float yy1 = std::max(input_boxes[i].box.y, input_boxes[j].box.y);
            float xx2 = std::min(input_boxes[i].box.width + input_boxes[i].box.x, input_boxes[j].box.width + input_boxes[j].box.x);
            float yy2 = std::min(input_boxes[i].box.height + input_boxes[i].box.y, input_boxes[j].box.height + input_boxes[j].box.y);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float h = std::max(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH)
            {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}