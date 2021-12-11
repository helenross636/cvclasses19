/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-18
 * @author Anonymous
 */

#include "cvlib.hpp"

namespace
{
struct descriptor : public std::vector<double>
{
    using std::vector<double>::vector;
    descriptor operator-(const descriptor& right) const
    {
        descriptor temp = *this;
        for (size_t i = 0; i < temp.size(); ++i)
        {
            temp[i] -= right[i];
        }
        return temp;
    }

    double norm_l1() const
    {
        double res = 0.0;
        for (auto v : *this)
        {
            res += std::abs(v);
        }
        return res;
    }
	double norm_l2() const
    {//ищем квадратный корень
        double res = 0.0;
        for (auto v : *this)
        {
            res += v * v;
        }
		return sqrt(res);
	}
};

void apply_filters(const cv::Mat& image, std::vector<cv::Mat>& kernels, std::vector<cv::Mat>& filtered_images)
{
	filtered_images.clear();
    for (auto& kernel : kernels)
    {
        cv::Mat response;
        cv::filter2D(image, response, CV_32F, kernel);
        filtered_images.push_back(response);
    }
}

void generate_kernels(std::vector<cv::Mat>& kernels, int kernel_size)
{
    kernels.clear();
    const double th = CV_PI / 4;
    const double lm = 10.0;
    const double gm = 0.5;
    auto lambdas = std::vector<int> {1,3,10,20};

    for (auto theta = CV_PI / 4; theta <= CV_PI; theta += CV_PI / 4)
    {
        for (auto sigma = 5; sigma <= 15; sigma += 5)
        {
            for (auto lambda: lambdas)
            {
                cv::Mat kernel = cv::getGaborKernel(cv::Size(kernel_size, kernel_size), sigma, theta, lambda, gm);
                kernels.push_back(kernel);
            }
        }
    }
}

void calculateDescriptor(const std::vector<cv::Mat>& filtered_images, const cv::Rect& roi, descriptor& descr)
{
    descr.clear();
    cv::Mat mean;
    cv::Mat dev;

    // \todo implement complete texture segmentation based on Gabor filters
    // (find good combinations for all Gabor's parameters)
	for (auto& response : filtered_images)
    {
		cv::meanStdDev(response(roi),mean,dev);		//взяли кусочек -> мат ож + дисперсия
        descr.emplace_back(mean.at<double>(0));		//добавили в дескриптор
        descr.emplace_back(dev.at<double>(0));
    }
}
}

namespace cvlib
{
cv::Mat select_texture(const cv::Mat& image, const cv::Rect& roi, double eps)
{
    cv::Mat imROI = image(roi);

    int kernel_size = std::min(roi.height, roi.width) / 2; // \todo round to nearest odd
    kernel_size = kernel_size % 2 ? kernel_size - 1 : kernel_size;
    std::vector<cv::Mat> kernels;
    generate_kernels(kernels, kernel_size);

    std::vector<cv::Mat> f_images;

    apply_filters(image, kernels, f_images);
	
    descriptor reference;	//область выделения
	
    calculateDescriptor(f_images, roi, reference);	//дескриптор выделенной	 области

    cv::Mat res = cv::Mat::zeros(image.size(), CV_8UC1);

    descriptor test(reference.size());
    cv::Rect baseROI = roi - roi.tl();	//сдвиг в левый верхний угол выделенного области

    // \todo move ROI smoothly pixel-by-pixel
    for (int i = 0; i < image.size().width - roi.width; i += 10)
    {
        for (int j = 0; j < image.size().height - roi.height; j += 10)
        {
            auto curROI = baseROI + cv::Point(i, j);
            calculateDescriptor(f_images,curROI, test);

            res(curROI) += 255 * ((test - reference).norm_l2() <= eps);	//из-за пересечения областей добавляем "+" (память)
		}
    }

    return res;
}
}
