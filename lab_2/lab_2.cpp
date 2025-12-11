#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>


cv::Mat calc_diff(cv::Mat img_1, cv::Mat img_2)
{
    cv::Mat diff_mat;

    cv::absdiff(img_1, img_2, diff_mat);
    
    cv::imshow("Difference between images", diff_mat);

    int non_zero_px = cv::countNonZero(diff_mat);

    return ((1.0 - (double)non_zero_px / (diff_mat.cols() * diff_mat.rows())) * 100.0)
}


cv::Mat box_filter_kernel(int size)
{
    cv::Mat kernel = cv::Mat::ones(size, size, CV_32F);
    kernel /= (float)(size * size);

    return kernel;
}


cv::Mat gauss_filter_kernel(int size)
{
    cv::Mat kernel(size, size, CV_32F);

    for (int x = 0; x < size; x++) {
        for (int y = 0; y < size; y++) {
            double sigma = 0.3 * ((size - 1) * 0.5 - 1.0) + 0.8;
            
        }
    }
}


int main()
{
    std::str task = "box_filter";
    cv::Mat src = "image.jpg";
    
    long tick_start = cv::getTickCount();

    switch (task)
    {
    case "box_filter":
        cv::Mat box_dst, blur_dst;

        long tick_start = cv::getTickCount();
        cv::filter2D(src, box_dst, box_filter_kernel(3));
        long time_end = (cv::getTickCount - time_start) / cv::getTickFrequency;
        std::cout << time_end;

        long tick_start = cv::getTickCount();
        cv::Mat blur = cv::blur(src, blur_dst, size);
        long time_end = (cv::getTickCount - time_start) / cv::getTickFrequency;
        std::cout << time_end;

        cv::imshow("Source image", src);
        cv::imshow("Filtered image", box_dst);
        cv::imshow("Filtered with blur image", blur);

        calc_diff(box_dst, blur_dst);
        cv::waitKey();
        cv::destroyAllWindows();

    }

    cv::waitKey();
    cv::destroyAllWindows();
}