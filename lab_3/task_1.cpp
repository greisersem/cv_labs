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
        cv::Scalar(0, 0, 255)
    );

    cv::line(
        img, 
        cv::Point(x, y - radius),
        cv::Point(x, y + radius),
        cv::Scalar(0, 0, 255)
    );
}


void allababah(cv::Mat img)
{
    cv::Mat src_gray;
    cv::cvtColor(img, src_gray, cv::COLOR_BGR2GRAY);
    cv::threshold(src_gray, src_gray, 230, 255, cv::THRESH_BINARY);
    
    std::vector <std::vector <cv::Point>> contours;
    cv::findContours(src_gray, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    std::vector <cv::Point> largest_contour = contours[0];

    for (int i = 0; i < contours.size(); i++) {
        if (contours[i].size() > largest_contour.size()) {
            largest_contour = contours[i];
        }
    }
    
    cv::Moments moments = cv::moments(largest_contour);
    int center_x = (moments.m10 / moments.m00);
    int center_y = (moments.m01 / moments.m00);

    draw_crosshair(img, center_x, center_y, 20);

    cv::imshow("Allababah", img);
    cv::waitKey();
}


int main()
{
    cv::Mat img_1 = cv::imread("/home/vboxuser/Desktop/cv_labs/lab_3/img_zadan/allababah/ig_0.jpg");
    cv::Mat img_2 = cv::imread("/home/vboxuser/Desktop/cv_labs/lab_3/img_zadan/allababah/ig_1.jpg");
    cv::Mat img_3 = cv::imread("/home/vboxuser/Desktop/cv_labs/lab_3/img_zadan/allababah/ig_2.jpg");
    allababah(img_1);
    allababah(img_2);
    allababah(img_3);
    cv::destroyAllWindows();
}