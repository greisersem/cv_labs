#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/intensity_transform.hpp>
#include <iostream>


void calc_diff(cv::Mat img_1, cv::Mat img_2)
{
    cv::Mat diff_mat, log_diff_mat;

    cv::absdiff(img_1, img_2, diff_mat);
    cv::imshow("Difference between images", diff_mat);

    cv::intensity_transform::logTransform(diff_mat, log_diff_mat);
    cv::imshow("Logarithm difference between images", diff_mat);

    int non_zero_px = cv::countNonZero(diff_mat);

    double s = (1.0 - (double)non_zero_px / (diff_mat.cols * diff_mat.rows)) * 100.0;
    std::cout << "Similarity " << s << "%" << std::endl;
}


cv::Mat box_filter_kernel(cv::Size2i kernel_size)
{
    cv::Mat kernel = cv::Mat::ones(kernel_size, CV_32F);
    kernel /= (float)(kernel_size.width * kernel_size.height);

    return kernel;
}

int main()
{
    cv::Mat src = cv::imread("250px-Lenna.png");
    cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
    cv::Size2i kernel_size(3, 3);

    cv::Mat box_dst, blur_dst;
/*
    double tick_start = cv::getTickCount();
    cv::filter2D(src, box_dst, -1, box_filter_kernel(kernel_size), cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
    double time_end = (double)(cv::getTickCount() - tick_start) / cv::getTickFrequency();
    std::cout << "Custom box: " << time_end << std::endl;

    tick_start = cv::getTickCount();
    cv::blur(src, blur_dst, kernel_size);
    time_end = (cv::getTickCount() - tick_start) / cv::getTickFrequency();
    std::cout << "Blur: "<< time_end << std::endl;

    cv::imshow("Source image", src);
    cv::imshow("Filtered image", box_dst);
    cv::imshow("Filtered with blur image", blur_dst);

    calc_diff(box_dst, blur_dst);
*/
    cv::Mat gauss_dst, diff_dst, log_diff_dst;

    double sigma = 1.0;

    double tick_start = cv::getTickCount();
    cv::blur(src, blur_dst, kernel_size);
    double time_box = (cv::getTickCount() - tick_start) / cv::getTickFrequency();

    tick_start = cv::getTickCount();
    cv::GaussianBlur(src, gauss_dst, kernel_size);
    double time_gauss = (cv::getTickCount() - tick_start) / cv::getTickFrequency();

    std::cout << "Box filter time: " << time_box << " s" << std::endl;
    std::cout << "Gauss filter time: " << time_gauss << " s" << std::endl;

    cv::imshow("Original", src);
    cv::imshow("Box Filter", blur_dst);
    cv::imshow("Gauss Filter", gauss_dst);
    calc_diff(blur_dst, gauss_dst);

    cv::waitKey();
    cv::destroyAllWindows();
}