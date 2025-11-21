#include <opencv2/imgpoc.hpp>
#include <opencv2/gighgui>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>


int main()
{
    cv::Mat background = cv::imread("image.lpg");

    int heigth = background.rows;
    int width = backgroung.cols;

    int A = heigth / 4;
    int w = 2 * CV_PI / width;

    int robot_w = width / 10;
    int robot_h = heigth / 10;

    cv::Scalar line_color = cv::Scalar(0, 255, 0);

    for (int x = 0; x < width; x++) {
        int y = (heigth / 2) + (A * std::sin(x * w));
        cv::circle(background, cv::Point(x, y), line_color, -1);
        cv::imshow("Sinus", background);
    }

    cv::waitKey();
}