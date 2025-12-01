#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>


int main()
{
    cv::Mat background = cv::imread("Untitled.jpeg");
    cv::String output_image = "output.jpeg";

    int heigth = background.rows;
    int width = background.cols;

    int A = heigth / 4;
    double w = 4 * CV_PI / width;

    int robot_w = width / 10;
    int robot_h = robot_w;
    cv::Scalar robot_color = cv::Scalar(255, 0, 0);

    cv::Scalar line_color = cv::Scalar(0, 255, 0);

    for (int x = 0; x < width; x++) {
        int y = (heigth / 2) + (A * std::sin(x * w + CV_PI));

        double derivative = A * w * std::cos(x * w + CV_PI);
        double angle = std::atan(derivative) * (180 / CV_PI);

        cv::circle(background, cv::Point(x, y), 5, line_color, -1);
        
        cv::Mat background_copy;
        background.copyTo(background_copy);
        
        cv::RotatedRect robot(
            cv::Point2f(x, y),
            cv::Size2f(robot_h, robot_w),
            angle
        );

        cv::Point2f vertices[4];
        robot.points(vertices);

        for (int i = 0; i < 4; i++){
            cv::line(background_copy, vertices[i], vertices[(i + 1) % 4], robot_color, 2);
        }

        cv::imshow("Robot moving by sinus", background_copy);
        cv::waitKey(10);
        if (x == width / 2) {
            cv::imwrite(output_image, background_copy);
        }
    }

    cv::destroyAllWindows();
}