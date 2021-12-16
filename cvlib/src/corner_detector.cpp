/* FAST corner detector algorithm implementation.
 * @file
 * @date 2018-10-16
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <ctime>

namespace cvlib
{
// static
auto genetate_pairs(int pairs_num, int S = 32, int s = 6, int seed = 3)
{
    srand(seed);
    std::vector<std::pair<std::pair<int, int>, int>> triplets;
    for (int i = 0; i < 2 * pairs_num; i++)
    {
        int anchor = rand() % (S - s) - (S / 2 - s / 2); //[-13; 13]
        int x = rand() % (S - s) - (S / 2 - s / 2);
        int y = rand() % (S - s) - (S / 2 - s / 2);
        triplets.push_back(std::make_pair((std::make_pair(x, y)), anchor));
    }
    return triplets;
}

cv::Ptr<corner_detector_fast> corner_detector_fast::create()
{
    return cv::makePtr<corner_detector_fast>();
}

bool check_fragment(cv::Mat& fragment)
{
    int M = 3;
    int N = 12;
    int threshold = 50;
    unsigned char I1 = std::min((int)fragment.at<unsigned char>(fragment.rows / 2, fragment.cols / 2) + threshold, 255);
    unsigned char I2 = std::max((int)fragment.at<unsigned char>(fragment.rows / 2, fragment.cols / 2) - threshold, 0);
    int i_ind[16] = {3, 6, 3, 0, 4, 5, 6, 6, 5, 4, 2, 1, 0, 0, 1, 2}; //начинаем со "среднего" пикселя
    int j_ind[16] = {6, 3, 0, 3, 6, 5, 4, 2, 1, 0, 0, 1, 2, 4, 5, 6}; //начинаем с "верхнего" пикселя
    int count1 = 0, count2 = 0;
    for (int k = 0; k < 4; k++)
    { //проверяем "крайние" пиксели
        if (fragment.at<unsigned char>(i_ind[k], j_ind[k]) > I1)
            count1++;

        if (fragment.at<unsigned char>(i_ind[k], j_ind[k]) < I2)
            count2++;
    }
    //проверка на угол
    if ((count1 < M) && (count2 < M))
        return false;

    for (int k = 4; k < 16; k++)
    { //нашли угол условно, убеждаемся
        if (fragment.at<unsigned char>(i_ind[k], j_ind[k]) > I1)
            count1++;
        if (fragment.at<unsigned char>(i_ind[k], j_ind[k]) < I2)
            count2++;
    }
    if (((count1 >= N) && (count2 < N)) ^ ((count1 < N) && (count2 >= N)))
        return true;
    else
        return false;
}

void corner_detector_fast::detect(cv::InputArray image, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray /*mask = cv::noArray()*/)
{
    keypoints.clear();
    cv::Mat curr_frame;
    image.getMat().copyTo(curr_frame);
    cv::cvtColor(curr_frame, curr_frame, cv::COLOR_BGR2GRAY);
    cv::medianBlur(curr_frame, curr_frame, 3); //медианное размытие

    int border = 3;
    //добавление границы
    cv::copyMakeBorder(curr_frame, curr_frame, border, border, border, border, cv::BORDER_REFLECT_101);

    for (int i = border; i < curr_frame.rows - border; i++)
        for (int j = border; j < curr_frame.cols - border; j++)
        {
            cv::Mat& fragment = curr_frame(cv::Range(i - border, i + border + 1), cv::Range(j - border, j + border + 1));
            if (check_fragment(fragment))
                keypoints.push_back(cv::KeyPoint(j, i, 2 * border + 1, 0, 0, 0, 3));
        }
    // \todo implement FAST with minimal LOCs(lines of code), but keep code readable.
}

void corner_detector_fast::compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
    // std::srand(unsigned(std::time(0))); // \todo remove me
    // \todo implement any binary descriptor
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY); //переводим в чб
    const int desc_length = 256; //длина дескриптора
    const int S = 32; //размер обрасти вокруг ключевой точки
    const int s = 6; //размер обрасти вокруг центра мини-патча
    std::vector<cv::KeyPoint> n_keypoints; //новый вектор ключевых точек (без крайних)
    for (int i = 0; i < keypoints.size(); i++)
    {
        auto point = keypoints[i];
        // point.pt - смещение; далее: лево, право, низ, верх
        if (((point.pt.x - S / 2) >= 0) && ((point.pt.x + S / 2) < gray.cols) && ((point.pt.y - S / 2) >= 0) && ((point.pt.y + S / 2) < gray.rows))
            n_keypoints.push_back(point);
    }
    std::cout << "deleted keypoints: " << keypoints.size() - n_keypoints.size() << std::endl;
	std::cout <<"TESTTEST" << std::endl;
    descriptors.create(static_cast<int>(n_keypoints.size()), desc_length, CV_32S);

    GaussianBlur(gray, gray, cv::Size(5, 5), 0, 0);

    auto triplets = genetate_pairs(desc_length * 2, S, s, 7);	//четная тройка - координаты х, нечетная - у
    auto desc_mat = descriptors.getMat();
    desc_mat.setTo(0);

    for (int i = 0; i < desc_mat.rows; i++)
    {
        auto curr_point = n_keypoints[i];
        for (int j = 0; j < desc_mat.cols; j++)
        {
            int x = 0, y = 0;
            //ищем расстояние по Хэммингу
            for (int Ox = -3; Ox <= 3; Ox++)
            {
                for (int Oy = (-3 + std::abs(Ox)); Oy <= (3 - std::abs(Ox)); Oy++)
                {
					int shift_x_x = triplets[j*2].first.first + Ox;
					int shift_x_y = triplets[j*2+1].first.first + Oy;
					int shift_y_x = triplets[j*2].first.second + Ox;
					int shift_y_y = triplets[j*2+1].first.second + Oy;
					int shift_an_x = triplets[j*2].second + Ox;
					int shift_an_y = triplets[j*2+1].second + Oy;
                    x += std::abs(gray.at<unsigned char>(curr_point.pt.x + shift_an_x, curr_point.pt.y + shift_an_y) - gray.at<unsigned char>(curr_point.pt.x + shift_x_x, curr_point.pt.y + shift_x_y));
					y += std::abs(gray.at<unsigned char>(curr_point.pt.x + shift_an_x, curr_point.pt.y + shift_an_y) - gray.at<unsigned char>(curr_point.pt.x + shift_y_x, curr_point.pt.y + shift_y_y));
                }
            }

            if (x * x > y * y)
                desc_mat.at<int>(i, j) = 1;
            else
                desc_mat.at<int>(i, j) = 0;
        }
    }

    /*
    descriptors.create(static_cast<int>(keypoints.size()), desc_length, CV_32S);
    auto desc_mat = descriptors.getMat();
    desc_mat.setTo(0);

    int* ptr = reinterpret_cast<int*>(desc_mat.ptr());
    for (const auto& pt : keypoints)
    {
        for (int i = 0; i < desc_length; ++i)
        {
            *ptr = std::rand();
            ++ptr;
        }
    }
*/
}

void corner_detector_fast::detectAndCompute(cv::InputArray, cv::InputArray, std::vector<cv::KeyPoint>&, cv::OutputArray descriptors, bool /*= false*/)
{
    // \todo implement me
}
} // namespace cvlib
