#include <iostream>
#include <vector>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


void draw_rectangle(cv::Mat img, cv::Point center, cv::Scalar color)
{
    int size = 60;
    cv::Rect rect(
        center.x - size / 2,
        center.y - size / 2,
        size,
        size
    );
    cv::rectangle(img, rect, color, 2);
}


cv::Mat find_nearest(cv::Mat img, cv::Point bulb_coords, std::string team, cv::Mat res)
{
    cv::Mat src_hsv;
    cv::cvtColor(img, src_hsv, cv::COLOR_BGR2HSV);

    cv::Scalar figures_color;
    if (team == "red") {
        cv::Mat temp_1, temp_2;
        cv::inRange(
            src_hsv,
            cv::Scalar(0, 120, 120),
            cv::Scalar(15, 255, 255),
            temp_1
        );
        cv::inRange(
            src_hsv,
            cv::Scalar(170, 120, 120),
            cv::Scalar(179, 255, 255),
            temp_2
        );
        src_hsv = temp_1 | temp_2;

        cv::circle(
            src_hsv, 
            cv::Point(bulb_coords.x - 5, bulb_coords.y - 40),
            35, 
            cv::Scalar(0,0,0),
            -1
        );

        figures_color = cv::Scalar(0, 0, 255);
    } else if (team == "blue")
    {
        cv::inRange(
            src_hsv,
            cv::Scalar(85, 120, 120),
            cv::Scalar(105, 255, 255),
            src_hsv
        );
        figures_color = cv::Scalar(255, 0, 0);
    } else if (team == "green")
    {
        cv::inRange(
            src_hsv,
            cv::Scalar(60, 80, 120),
            cv::Scalar(80, 255, 255),
            src_hsv
        );
        figures_color = cv::Scalar(0, 255, 0);
    }

    std::vector <std::vector <cv::Point>> contours;
    cv::findContours(src_hsv, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    std::vector <cv::Point> largest_contour = contours[0];

    for (int i = 0; i < contours.size(); i++) {
        if (contours[i].size() > largest_contour.size()) {
            largest_contour = contours[i];
        }
    }

    double min_dist = 1e6;
    cv::Point best_center;

    for (int i = 0; i < contours.size(); i++) {
        if (contours[i].size() > largest_contour.size() * 0.3) {
            cv::Moments moments = cv::moments(contours[i]);
            int center_x = (moments.m10 / moments.m00);
            int center_y = (moments.m01 / moments.m00);
            draw_rectangle(res, cv::Point(center_x, center_y), figures_color);
            
            double distance_to_bulb = 
                sqrt(pow(center_x - bulb_coords.x, 2) + pow(center_y - bulb_coords.y, 2));
            if (distance_to_bulb < min_dist) {
                min_dist = distance_to_bulb;
                best_center = cv::Point(center_x, center_y);
            }
        }
    }

    cv::line(res, best_center, bulb_coords, figures_color, 2);
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
    cv::Mat res = img.clone();
    cv::Point bulb_coords = find_bulb(img);
    draw_rectangle(res, bulb_coords, cv::Scalar(0, 0, 0));

    int red_team_hue = 10;
    int blue_team_hue = 95;
    int green_team_hue = 70;

    cv::Mat red = find_nearest(img, bulb_coords, "red", res);
    cv::imshow("red", red);
    cv::Mat blue = find_nearest(img, bulb_coords, "blue", res);
    cv::imshow("blue", blue);
    cv::Mat green = find_nearest(img, bulb_coords, "green", res);
    cv::imshow("green", green);
    cv::imshow("img", res);
}


int main()
{
    cv::Mat img = cv::imread("/home/vboxuser/Desktop/cv_labs/lab_3/img_zadan/roboti/roi_robotov.jpg");
    cv::Mat img_1 = cv::imread("/home/vboxuser/Desktop/cv_labs/lab_3/img_zadan/roboti/roi_robotov_1.jpg");
    robots_detection(img);
    cv::waitKey();
    robots_detection(img_1);
    cv::waitKey();
    cv::destroyAllWindows();
}