#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


std::vector<int> find_nearest(cv::Mat img, cv::Point bulb_coords, int robot_team_hue)
{
    cv::Mat temp = img.clone();

    cv::inRange(
        temp,
        cv::Scalar(robot_team_hue - 10, 0, 0),
        cv::Scalar(robot_team_hue + 10, 255, 255),
        temp
    );

    cv::imshow("img", temp);
    cv::waitKey();
    cv::destroyAllWindows();
}


cv::Point find_bulb(cv::Mat img)
{
    cv::Mat bulb;
    cv::Scalar bulb_min(0, 0, 250);
    cv::Scalar bulb_max(150, 30, 255);

    cv::inRange(
        img, 
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
    cv::Mat src_hsv;
    cv::cvtColor(img, src_hsv, cv::COLOR_BGR2HSV);

    cv::Point bulb_coords = find_bulb(src_hsv);
    int size = 40;
    cv::Rect rect(
        bulb_coords.x - size / 2,
        bulb_coords.y - size / 2,
        size,
        size
    );
    cv::rectangle(img, rect, cv::Scalar(0, 0, 0));

    int red_team_hue = 172;
    int blue_team_hue = 95;
    int green_team_hue = 70;

    find_nearest(src_hsv, bulb_coords, red_team_hue);
    find_nearest(src_hsv, bulb_coords, blue_team_hue);
    find_nearest(src_hsv, bulb_coords, green_team_hue);
}


int main()
{
    cv::Mat img = cv::imread("/home/vboxuser/Desktop/cv_labs/lab_3/img_zadan/roboti/roi_robotov_1.jpg");
    robots_detection(img);
}