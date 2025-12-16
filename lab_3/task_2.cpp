#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void draw_crosshair(cv::Mat img, int x, int y, int radius)
{
    cv::circle(
        img, 
        cv::Point(x, y),
        radius, 
        cv::Scalar(0, 0, 255),
        3
    );
    
    cv::line(
        img, 
        cv::Point(x - radius, y), 
        cv::Point(x + radius, y), 
        cv::Scalar(0, 0, 255),
        2
    );

    cv::line(
        img, 
        cv::Point(x, y - radius),
        cv::Point(x, y + radius),
        cv::Scalar(0, 0, 255),
        2
    );
}


void teplovizor(cv::Mat img)
{
    cv::Mat src_hsv;
    cv::cvtColor(img, src_hsv, cv::COLOR_BGR2HSV);

    int min_hue_thresh = 0, min_sat_thresh = 50, min_val_thresh = 50;
    int max_hue_thresh = 37, max_sat_thresh = 255, max_val_thresh = 255;
    
    cv::inRange(
        src_hsv, 
        cv::Scalar(min_hue_thresh, min_sat_thresh, min_val_thresh),
        cv::Scalar(max_hue_thresh, max_sat_thresh, max_val_thresh),
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
        if (contours[i].size() > largest_contour.size() * 0.8) {
            cv::Moments moments = cv::moments(contours[i]);
            int center_x = (moments.m10 / moments.m00);
            int center_y = (moments.m01 / moments.m00);

            draw_crosshair(img, center_x, center_y, 20);
        }
    }

    cv::imshow("img", img);
    cv::waitKey();
}


int main()
{
    cv::Mat img_1 = cv::imread("/home/vboxuser/Desktop/cv_labs/lab_3/img_zadan/teplovizor/21331.res.jpg");
    cv::Mat img_2 = cv::imread("/home/vboxuser/Desktop/cv_labs/lab_3/img_zadan/teplovizor/445923main_STS128_16FrAvg_color1.jpg");
    cv::Mat img_3 = cv::imread("/home/vboxuser/Desktop/cv_labs/lab_3/img_zadan/teplovizor/MW-AW129-measured.jpg");
    cv::Mat img_4 = cv::imread("/home/vboxuser/Desktop/cv_labs/lab_3/img_zadan/teplovizor/ntcs_quest_measurement.png");
    cv::Mat img_5 = cv::imread("/home/vboxuser/Desktop/cv_labs/lab_3/img_zadan/teplovizor/size0-army.mil-2008-08-28-082221.jpg");

    teplovizor(img_1);
    teplovizor(img_2);
    teplovizor(img_3);
    teplovizor(img_4);
    teplovizor(img_5);
    cv::destroyAllWindows();
}