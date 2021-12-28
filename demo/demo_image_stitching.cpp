/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-11-25
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>

#include "utils.hpp"

int demo_image_stitching(int argc, char* argv[])
{
    // \todo implenent demo
	cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    const auto main_wnd = "orig";
    const auto demo_wnd = "demo";

    cv::namedWindow(main_wnd);
    cv::namedWindow(demo_wnd);

    struct img_features
    {
        cv::Mat img;
        std::vector<cv::KeyPoint> corners;
        cv::Mat descriptors;
    };

    img_features ref;
    img_features test;


    cv::Mat main_frame;
    cv::Mat demo_frame;
    utils::fps_counter fps;
    int pressed_key = 0;

    auto stitcher = cvlib::Stitcher::create();

    while (pressed_key != 27) // ESC
    {
        cap >> test.img;
        cv::imshow(main_wnd, test.img);

        pressed_key = cv::waitKey(30);
        if (pressed_key == ' ') // space
        {
            ref.img = test.img.clone();
        }
        if(!ref.img.empty())
        {
            auto result = stitcher->Stiched(test.img,ref.img);
            cv::imshow(demo_wnd, result);
        }
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_wnd);
    return 0;
}
