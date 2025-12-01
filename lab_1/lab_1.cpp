#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>


int main()
{
    cv::Mat background = cv::imread("250px-Lenna.png");

    int heigth = background.rows;
    int width = background.cols;

    int A = heigth / 4;
    double w = 2 * CV_PI / width;

    int robot_w = width / 10;
    int robot_h = heigth / 10;
    cv::Scalar robot_color = cv::Scalar(255, 0, 0);

    cv::Scalar line_color = cv::Scalar(0, 255, 0);

    for (int x = 0; x < width; x++) {
        int y = (heigth / 2) + (A * std::sin(x * w));
        cv::circle(background, cv::Point(x, y), 5, line_color, -1);
        
        cv::Mat background_copy;
        background.copyTo(background_copy);
        cv::rectangle(background_copy, cv::Point(x, y), cv::Point(x + robot_w, y + robot_h));
        cv::imshow("Robot moving by sinus", background_copy);
        cv::waitKey(1);
    }

    cv::waitKey();
}