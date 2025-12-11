#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>


cv::Mat box_filter_kernel(int size)
{
    cv::Mat kernel = cv::Mat::ones(size, size, CV_32F);
    kernel /= (float)(size * size);

    return kernel;
}


int main()
{
    cv::Mat src = "image.jpg";
    cv::Mat dst;
    
    long tick_start = cv::getTickCount();

    cv::filter2D(src, dst, box_filter_kernel(3));
    cv::imshow("Filtered image", dst);

    long time_end = (cv::getTickCount - time_start) / cv::getTickFrequency;

    cv::waitKey();
    cv::destroyAllWindows();
}