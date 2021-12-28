#include "cvlib.hpp"
#include <opencv2/opencv.hpp>

namespace cvlib
{
cv::Mat Stitcher::Stiched(cv::Mat img1, cv::Mat img2)
{
    auto detector = cv::AKAZE::create();	//дескрипт
    auto matcher = cvlib::descriptor_matcher(0.7f);	//сопроставляем
    struct img_features
    {
        cv::Mat img;
        std::vector<cv::KeyPoint> corners;
        cv::Mat descriptors;
    };

    img_features test;	//видео
    img_features ref;	//пробел

	img1.copyTo(test.img);	//видео
	img2.copyTo(ref.img);	//пробел

    std::vector<std::vector<cv::DMatch>> pairs;

    detector->detectAndCompute(test.img, cv::Mat(), test.corners, test.descriptors);
    detector->detectAndCompute(ref.img, cv::Mat(), ref.corners, ref.descriptors);
    matcher.radiusMatch(test.descriptors, ref.descriptors, pairs, 20000);

    std::vector<cv::Point2f> obj, scene;	//вектор точек (текущий, с пробела)
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < pairs.size(); ++i)
    {
        if (pairs[i].size())
			good_matches.push_back((pairs[i][0]));
    }
    for (size_t i = 0; i < good_matches.size(); i++)
    {
        obj.push_back(test.corners[good_matches[i].queryIdx].pt);
        scene.push_back(ref.corners[good_matches[i].trainIdx].pt);
    }

    cv::Rect croppImg1(0, 0, img1.cols, img1.rows);	//для обрезки изобр
    cv::Rect croppImg2(0, 0, img2.cols, img2.rows);
    int movementDirection = 0;	//спещение по вертикали

	if (obj[0].x >= scene[0].x)
	{//"живое" слева?
		croppImg1.width = obj[0].x;	//обрезаем живое до точки (справа)
        croppImg2.x = scene[0].x;	//двигаем срез
        croppImg2.width = img2.cols - croppImg2.x;	//обрезаем пробел (слева) 
    }
	else
	{//наоборот
		croppImg1.width = croppImg1.width - obj[0].x;	//правая часть от живого
		croppImg1.x = obj[0].x;	//двигаем срез
		croppImg2.width = scene[0].x;	//левая часть от пробела
	}
	movementDirection = obj[0].y - scene[0].y;

    img1 = img1(croppImg1);	//обрезаем
    img2 = img2(croppImg2);	//обрезаем
    int maxHeight = (img1.rows < img2.rows) ? img1.rows : img2.rows;	//выбираем высоту по минимальному
    int maxWidth = img1.cols + img2.cols;	//находим общую ширину
    cv::Mat result=cv::Mat::zeros(cv::Size(maxWidth, maxHeight + abs(movementDirection)), CV_8UC3);	//готовимся!

	if (obj[0].x < scene[0].x)
	{//работаем на оба фронта
		cv::Mat temp_img;
		img1.copyTo(temp_img);
		img2.copyTo(img1);
		temp_img.copyTo(img2);
    }
	//учитываем смещение по вертикали
    if (movementDirection > 0)
    {
		cv::Mat half1(result, cv::Rect(0, 0, img1.cols, img1.rows));
		img1.copyTo(half1);
		cv::Mat half2(result, cv::Rect(img1.cols, abs(movementDirection),img2.cols, img2.rows));
		img2.copyTo(half2);
    }
    else
    {
		cv::Mat half1(result, cv::Rect(0, abs(movementDirection), img1.cols, img1.rows));
		img1.copyTo(half1);
		cv::Mat half2(result, cv::Rect(img1.cols,0 ,img2.cols, img2.rows));
		img2.copyTo(half2);
    }
    return result;
}
};
