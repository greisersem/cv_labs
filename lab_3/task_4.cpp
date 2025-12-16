#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void gk(cv::Mat img, cv::Mat templ) {
    cv::Mat res, templ_res;

    cv::cvtColor(templ, templ_res, cv::COLOR_BGR2GRAY);
    cv::threshold(templ_res, templ_res, 240, 255, cv::THRESH_BINARY_INV);

    std::vector <std::vector <cv::Point>> templ_wrench_contour;
    cv::findContours(templ_res, templ_wrench_contour, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    cv::Moments mnts = cv::moments(templ_wrench_contour[0]);
    double m00 = mnts.m00;
    double m10 = mnts.m10;
    double m01 = mnts.m01;

    double templ_area = cv::contourArea(templ_wrench_contour[0]);
    double templ_perimeter = cv::arcLength(templ_wrench_contour[0], true);

    cv::cvtColor(img, res, cv::COLOR_BGR2GRAY);
    cv::threshold(res, res, 230, 255, cv::THRESH_BINARY_INV);
    std::vector <std::vector <cv::Point>> wrenchs_contour;
    cv::findContours(res, wrenchs_contour, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    for (int i = 0; i < wrenchs_contour.size(); i++) {
        cv::Moments mnts = cv::moments(wrenchs_contour[i]);
        double area = cv::contourArea(wrenchs_contour[i]);
        double perimeter = cv::arcLength(wrenchs_contour[i], true);
        double ratio = area / (perimeter * perimeter);
        double templ_ratio = templ_area / (templ_perimeter * templ_perimeter);
        std::cout << i << " match: " << ratio << std::endl;
        std::cout << i << " match: " << templ_ratio << std::endl;
        if (std::abs(templ_ratio - ratio) < 0.019) {  
            cv::polylines(img, wrenchs_contour[i], true, cv::Scalar(0, 255, 0), 5, 8);
            cv::putText(
                img, 
                "YES", 
                cv::Point(mnts.m10 / mnts.m00 - 60 , mnts.m01 / mnts.m00 - 60), 
                cv::FONT_HERSHEY_COMPLEX, 0.6, 
                cv::Scalar(0, 255, 0), 
                1.0
            );
        }
        else
        {
            cv::polylines(img, wrenchs_contour[i], true, cv::Scalar(0, 0, 255), 5, 8);
            cv::putText(
                img, 
                "NO", 
                cv::Point(mnts.m10 / mnts.m00 - 40 , mnts.m01 / mnts.m00 - 50), 
                cv::FONT_HERSHEY_COMPLEX, 0.6, 
                cv::Scalar(0, 0, 255), 
                1.0
            );
        }
    }

    cv::imshow("src", img);
    cv::imshow("res", res);
    cv::imshow("templ", templ);
    cv::waitKey(-1);
}


int main()
{
    cv::Mat img = cv::imread("/home/vboxuser/Desktop/cv_labs/lab_3/img_zadan/gk/gk.jpg");
    cv::Mat tmpl = cv::imread("/home/vboxuser/Desktop/cv_labs/lab_3/img_zadan/gk/gk_tmplt.jpg");

    gk(img, tmpl);
}