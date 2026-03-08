#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

void get_laser(const cv::Mat &frame, cv::Mat &binarized_img)
{
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, gray, 180, 255, cv::THRESH_BINARY);
    cv::blur(gray, gray, cv::Size(3, 3));
    cv::Canny(gray, binarized_img, 100, 150);
}


void calibration()
{
    cv::Mat calib_img = cv::imread("/home/vboxuser/Desktop/cv_labs/lab_6/Video/calib_1_0.jpg");
    cv::Mat lines_img = calib_img.clone();

    cv::Mat binarized_img;
    std::vector<cv::Vec2f> lines;
    get_laser(calib_img, binarized_img);
    cv::imshow("binarized", binarized_img);
    cv::waitKey();
    cv::HoughLines(binarized_img, lines, 1.0, CV_PI / 360, 37);
    std::cout << "lines found: " << lines.size() << std::endl;

    for (int i = 0; i < lines.size(); i++) {
        float rho = lines[i][0], theta = lines[i][1];
        cv::Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        cv::line(lines_img, pt1, pt2, cv::Scalar(0,0,255), 1);

    cv::imshow("lines", lines_img);
    cv::waitKey();
    }
}


int main()
{
    const int CAM_ANGLE_X = 74;  // deg
    const float CAM_ANGLE_X_RAD = CAM_ANGLE_X * CV_PI / 180.0f; //rad
    const int Y = -250;  // mm
    const float scale = 0.2f;

    calibration();
    cv::VideoCapture cap("/home/vboxuser/Desktop/cv_labs/lab_6/Video/2.avi");

    int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    cv::Point2i center(width / 2, height / 2);

    const float fx = (width / 2.0f) / tan(CAM_ANGLE_X_RAD / 2.0f);
    const float fy = fx;

    cv::Mat frame, binarized_img;

    while (cap.read(frame)) {
        cv::Mat map = cv::Mat::zeros(height, width, CV_8UC3);

        for (int i = 0; i < width; i += 25) {
            cv::line(map, cv::Point(i, 0), cv::Point(i, height), cv::Scalar(255, 255, 255));
            if (i < height) {
                cv::line(map, cv::Point(0, i), cv::Point(width, i), cv::Scalar(255, 255, 255));
            }
        }

        get_laser(frame, binarized_img);

        for (int pix_y = 0; pix_y < height; pix_y++) {
            for (int pix_x = 0; pix_x < width; pix_x++) {
                if (binarized_img.at<uchar>(pix_y, pix_x) == 255) {
                    double x_foc = (center.x - pix_x) / fx;
                    double y_foc = (center.y - pix_y) / fy;

                    if (std::abs(y_foc) < 1e-6) continue;

                    double k = Y / y_foc;

                    double x_real = k * x_foc;
                    double z_real = k;

                    int map_x = static_cast<int>(x_real * scale) + width / 2;
                    int map_y = static_cast<int>(z_real * scale);

                    if (map_x >= 0 && map_x < width &&
                        map_y >= 0 && map_y < height) {
                        cv::circle(map, cv::Point(map_x, map_y), 1, cv::Scalar(0, 255, 0), -1);
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
}