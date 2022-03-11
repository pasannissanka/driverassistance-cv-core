#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

int main(int, char **)
{
    Mat image;
    namedWindow("Display window");
    VideoCapture cap(0);

    if (!cap.isOpened())
    {
        std::cout << "cannot open camera" << std::endl;
    }

    while (true)
    {
        cap >> image;
        imshow("Display window", image);

        // Press ESC to exit
        if ((char)waitKey(25) == 27)
        {
            break;
        }
    }

    return 0;
}
