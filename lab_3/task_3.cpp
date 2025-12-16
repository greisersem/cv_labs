#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


void draw_rectangle(cv::Mat img, cv::Point center, cv::Scalar color)
{
    int size = 40;
    cv::Rect rect(
        center.x - size / 2,
        center.y - size / 2,
        size,
        size
    );
    cv::rectangle(img, rect, color, 2);
}


cv::Mat find_nearest(cv::Mat img, cv::Point bulb_coords, int robot_team_hue)
{
    cv::Mat src_hsv;
    cv::cvtColor(img, src_hsv, cv::COLOR_BGR2HSV);

    int h_min = std::max(0, robot_team_hue - 10);
    int h_max = std::min(179, robot_team_hue + 10);

    cv::inRange(
        src_hsv,
        cv::Scalar(h_min, 120, 120),
        cv::Scalar(h_max, 255, 255),
        src_hsv
    );

    std::vector <std::vector <cv::Point>> contours;
    cv::findContours(src_hsv, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    std::vector <cv::Point> largest_contour = contours[0];

    for (int i = 0; i < contours.size(); i++) {
        if (contours[i].size() > largest_contour.size()) {
            largest_contour = contours[i];
        }
    }

    for (int i = 0; i < contours.size(); i++) {
        if (contours[i].size() > largest_contour.size() * 0.1) {
            cv::Moments moments = cv::moments(contours[i]);
            int center_x = (moments.m10 / moments.m00);
            int center_y = (moments.m01 / moments.m00);
            draw_rectangle(img, cv::Point(center_x, center_y), cv::Scalar(0,0,0));
        }
    }

    return src_hsv;
}


cv::Point find_bulb(cv::Mat img)
{
    cv::Mat bulb;
    cv::cvtColor(img, bulb, cv::COLOR_BGR2HSV);
    cv::Scalar bulb_min(0, 0, 250);
    cv::Scalar bulb_max(150, 10, 255);

    cv::inRange(
        bulb, 
        bulb_min,
        bulb_max, 
        bulb
    );

    std::vector <std::vector <cv::Point>> contours;
    cv::findContours(bulb, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    std::vector <cv::Point> largest_contour = contours[0];

    for (int i = 0; i < contours.size(); i++) {
        if (contours[i].size() > largest_contour.size()) {
            largest_contour = contours[i];
        }
    }
    
    cv::Moments moments = cv::moments(largest_contour);
    int center_x = (moments.m10 / moments.m00);
    int center_y = (moments.m01 / moments.m00);

    return cv::Point(center_x, center_y);
}


void robots_detection(cv::Mat img)
{
    cv::Point bulb_coords = find_bulb(img);
    draw_rectangle(img, bulb_coords, cv::Scalar(0, 0, 0));

    int red_team_hue = 10;
    int blue_team_hue = 95;
    int green_team_hue = 70;

    cv::Mat red = find_nearest(img, bulb_coords, red_team_hue);
    cv::imshow("red", red);
    cv::Mat blue = find_nearest(img, bulb_coords, blue_team_hue);
    cv::imshow("blue", blue);
    cv::Mat green = find_nearest(img, bulb_coords, green_team_hue);
    cv::imshow("green", green);
    cv::imshow("img", img);
    cv::waitKey();
}


int main()
{
    cv::Mat img = cv::imread("/home/vboxuser/Desktop/cv_labs/lab_3/img_zadan/roboti/roi_robotov_1.jpg");
    robots_detection(img);
    cv::destroyAllWindows();
}