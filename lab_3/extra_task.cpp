#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>


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


cv::Mat calc_hist(const cv::Mat& roi) 
{
    cv::Mat hsv;
    cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);

    int h_bins = 30; 
    int s_bins = 32;
    int hist_size[] = {h_bins, s_bins};

    float h_range[] = {0, 180};
    float s_range[] = {0, 256};
    const float* ranges[] = {h_range, s_range};

    int channels[] = {0, 1};
    cv::Mat hist;
    cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 2, hist_size, ranges, true, false);
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);
    return hist;
}


void detect_robots(const cv::Mat& img, const cv::Mat& hist_red, const cv::Mat& hist_blue, const cv::Mat& hist_green, cv::Point bulb_coords)
{
    cv::Mat res = img.clone();

    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsv_channels;
    cv::split(hsv, hsv_channels);

    cv::Mat mask;
    cv::threshold(hsv_channels[1], mask, 60, 255, cv::THRESH_BINARY);
    cv::circle(
            mask, 
            cv::Point(bulb_coords.x - 5, bulb_coords.y - 40),
            35, 
            cv::Scalar(0,0,0),
            -1
        );

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    cv::imshow("contours", mask);

    for (auto& contour : contours) {
        if (cv::contourArea(contour) < 100) continue;

        cv::Rect bbox = cv::boundingRect(contour);
        cv::Mat obj_roi = img(bbox);

        cv::Mat obj_hist = calc_hist(obj_roi);

        double sim_red = cv::compareHist(obj_hist, hist_red, cv::HISTCMP_CORREL);
        double sim_blue = cv::compareHist(obj_hist, hist_blue, cv::HISTCMP_CORREL);
        double sim_green = cv::compareHist(obj_hist, hist_green, cv::HISTCMP_CORREL);

        double max_sim = std::max({sim_red, sim_blue, sim_green});

        if (max_sim < 0.01) continue;

        cv::Scalar color;
        if (max_sim == sim_red) color = cv::Scalar(0, 0, 255);
        else if (max_sim == sim_blue) color = cv::Scalar(255, 0, 0);
        else color = cv::Scalar(0, 255, 0);

        cv::Moments m = cv::moments(contour);
        cv::Point center(m.m10 / m.m00, m.m01 / m.m00);

        draw_rectangle(res, center, color);
    }

    cv::imshow("robots_classified", res);
}


int main()
{
    cv::Mat img = cv::imread("/home/vboxuser/Desktop/cv_labs/lab_3/img_zadan/roboti/roi_robotov.jpg");

    cv::Mat red_roi = cv::imread("/home/vboxuser/Desktop/cv_labs/lab_3/img_zadan/roboti/red.png");
    cv::Mat blue_roi = cv::imread("/home/vboxuser/Desktop/cv_labs/lab_3/img_zadan/roboti/blue.png");
    cv::Mat green_roi = cv::imread("/home/vboxuser/Desktop/cv_labs/lab_3/img_zadan/roboti/green.png");

    cv::Point bulb_coords = find_bulb(img);

    cv::Mat hist_red = calc_hist(red_roi);
    cv::Mat hist_blue = calc_hist(blue_roi);
    cv::Mat hist_green = calc_hist(green_roi);

    cv::VideoCapture cap("/home/vboxuser/Desktop/cv_labs/lab_3/img_zadan/roboti/Robot Swarm.mp4");

    int count = 0;
    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (!frame.empty()) {
            detect_robots(frame, hist_red, hist_blue, hist_green, bulb_coords);
            count++;
            cv::waitKey(10);
        } else {
            break;
        }
    }
    cap.release();
    return 0;
}

