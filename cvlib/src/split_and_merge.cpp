/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include "cvlib.hpp"

namespace
{
void split_image(cv::Mat image, double stddev)
{
    cv::Mat mean;
    cv::Mat dev;
    cv::meanStdDev(image, mean, dev);

    if (dev.at<double>(0) <= stddev)
    {
        image.setTo(mean);
        return;
    }

    const auto width = image.cols;
    const auto height = image.rows;
	
	if ((height != 1) && (width != 1))
    {
		split_image(image(cv::Range(0, height / 2), cv::Range(0, width / 2)), stddev);
		split_image(image(cv::Range(0, height / 2), cv::Range(width / 2, width)), stddev);
		split_image(image(cv::Range(height / 2, height), cv::Range(width / 2, width)), stddev);
		split_image(image(cv::Range(height / 2, height), cv::Range(0, width / 2)), stddev);
	}
}
} // namespace

void merge_image(cv::Mat image, double stddev)
{
	uint8_t meanres;
	for (int i = 0; i < image.rows-1; i++)
	{//ходим по строкам
		for (int j = 0; j < image.cols-1; j++)
		{//ходим по столбцам
			if (abs(image.at<uint8_t>(i, j) - image.at<uint8_t>(i + 1, j + 1)) <= stddev)
			{//сравнение по диагонали
				meanres = (image.at<uint8_t>(i, j) + image.at<uint8_t>(i + 1, j + 1)) / 2;
				image.at<uint8_t>(i, j) = meanres;
				image.at<uint8_t>(i + 1, j + 1) = meanres;
			}
			if (abs(image.at<uint8_t>(i, j) - image.at<uint8_t>(i + 1, j)) <= stddev)
			{//сравнение по горизонтали
				meanres = (image.at<uint8_t>(i, j) + image.at<uint8_t>(i + 1, j)) / 2;
				image.at<uint8_t>(i, j) = meanres;
				image.at<uint8_t>(i + 1, j) = meanres;
			}
			if (abs(image.at<uint8_t>(i, j) - image.at<uint8_t>(i, j + 1)) <= stddev)
			{//сравнение по вертикали 
				meanres = (image.at<uint8_t>(i, j) + image.at<uint8_t>(i, j + 1)) / 2;
				image.at<uint8_t>(i, j) = meanres;
				image.at<uint8_t>(i, j + 1) = meanres;
			}
		}
	}
}

namespace cvlib
{
cv::Mat split_and_merge(const cv::Mat& image, double stddev)
{
    // split part
    cv::Mat res = image;
    split_image(res, stddev);

    // merge part
    merge_image(res, stddev);
    return res;
}
} // namespace cvlib
