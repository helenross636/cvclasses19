/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-11-25
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>

#include "utils.hpp"

int demo_feature_matching(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    const auto main_wnd = "orig";
    const auto demo_wnd = "demo";

    cv::namedWindow(main_wnd);
    cv::namedWindow(demo_wnd);

    //auto detector = cv::AKAZE::create(); // \todo use your detector from cvlib
    auto detector = cvlib::corner_detector_fast().create();
    auto matcher = cvlib::descriptor_matcher(1.2f); //\todo add trackbar to demo_wnd to tune ratio value

    /// \brief helper struct for tidy code
    struct img_features
    {
        cv::Mat img;
        std::vector<cv::KeyPoint> corners;
        cv::Mat descriptors;
    };
	cv::Mat temp_img;
    img_features ref;	//space
    img_features test;
    std::vector<std::vector<cv::DMatch>> pairs;

    cv::Mat main_frame;
    cv::Mat demo_frame;
    utils::fps_counter fps;
    int pressed_key = 0;

    int ratio = 4;
    int max_distance = 45;
    int cornet_thresh = 200;
    cv::createTrackbar("ratio", demo_wnd, &ratio, 15);
    cv::createTrackbar("max distance", demo_wnd, &max_distance, 150);
    cv::createTrackbar("thresh for cornet distance", demo_wnd, &cornet_thresh, 300);
    while (pressed_key != 27) // ESC
    {
        cap >> test.img;
		std::cout << "test1" << std::endl;

        detector->thresh = cornet_thresh;
        detector->detect(test.img, test.corners);
        cv::drawKeypoints(test.img, test.corners, main_frame);
        cv::imshow(main_wnd, main_frame);
		std::cout << "test2" << std::endl;

        pressed_key = cv::waitKey(30);
        if (pressed_key == ' ') // space
        {
			std::cout << "SPACE" << std::endl;
            ref.img = test.img.clone();
            detector->detectAndCompute(ref.img, cv::Mat(), ref.corners, ref.descriptors);
        }

        if (ref.corners.empty())
        {
            continue;
        }

        matcher.set_ratio(ratio/10.0f);
		test.img.copyTo(temp_img);
		cv::GaussianBlur(temp_img, temp_img, cv::Size(5, 5), 0, 0);
        detector->detectAndCompute(test.img, cv::Mat(), test.corners, test.descriptors);
        //\todo add trackbar to demo_wnd to tune threshold value
        matcher.radiusMatch(test.descriptors, ref.descriptors, pairs, max_distance);
        cv::drawMatches(test.img, test.corners, ref.img, ref.corners, pairs, demo_frame);

        utils::put_fps_text(demo_frame, fps);
        cv::imshow(demo_wnd, demo_frame);
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_wnd);

    return 0;
}