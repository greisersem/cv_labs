#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

void get_laser(const cv::Mat &frame, cv::Mat &binarized_img)
{
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(
        hsv,
        cv::Scalar(0, 0, 90),
        cv::Scalar(93, 115, 255),
        binarized_img 
    );
}

int main()
{
    const int CAM_ANGLE = 74;  // deg
    const float CAM_ANGLE_RAD = CAM_ANGLE * CV_PI / 180.0f; //rad
    const int Y = -250;  // mm
    const float scale = 0.2f;

    cv::VideoCapture cap("/home/vboxuser/Desktop/cv_labs/lab_6/Video/calib_1.avi");

    if (!cap.isOpened()) {
        std::cout << "Ошибка открытия видео!" << std::endl;
        return -1;
    }

    int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    cv::Point2i center(width / 2, height / 2);

    const float focus_x = (width / 2.0f) / tan(CAM_ANGLE_RAD / 2.0f);
    const float focus_y = (height / 2.0f) / tan(CAM_ANGLE_RAD / 2.0f);

    cv::Mat frame, binarized_img;

    while (cap.read(frame))
    {
        cv::Mat map = cv::Mat::zeros(height, width, CV_8UC3);

        for (int i = 0; i < width; i += 25)
            cv::line(map, cv::Point(i, 0), cv::Point(i, height), cv::Scalar(255, 255, 255));

        for (int i = 0; i < height; i += 25)
            cv::line(map, cv::Point(0, i), cv::Point(width, i), cv::Scalar(255, 255, 255));

        get_laser(frame, binarized_img);

        for (int pix_y = 0; pix_y < height; pix_y++)
        {
            for (int pix_x = 0; pix_x < width; pix_x++)
            {
                if (binarized_img.at<uchar>(pix_y, pix_x) == 255)
                {
                    double x_foc = (center.x - pix_x) / focus_x;
                    double y_foc = (center.y - pix_y) / focus_y;

                    if (std::abs(y_foc) < 1e-6) continue;

                    double k = Y / y_foc;

                    double x_real = k * x_foc;
                    double z_real = k;

                    int map_x = static_cast<int>(x_real * scale) + width / 2;
                    int map_y = static_cast<int>(z_real * scale);

                    if (map_x >= 0 && map_x < width &&
                        map_y >= 0 && map_y < height)
                    {
                        cv::circle(map, cv::Point(map_x, map_y), 1, cv::Scalar(0, 255, 0));
                    }
                }
            }
        }

        cv::imshow("Video", frame);
        cv::imshow("Laser mask", binarized_img);
        cv::imshow("Map", map);

        if (cv::waitKey(30) == 27)
            break;
    }

    return 0;
}