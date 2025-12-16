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

    cv::cvtColor(img, res, cv::COLOR_BGR2GRAY);
    cv::threshold(res, res, 230, 255, cv::THRESH_BINARY_INV);
    std::vector <std::vector <cv::Point>> wrenchs_contour;
    cv::findContours(res, wrenchs_contour, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    for (int i = 0; i < wrenchs_contour.size(); i++) {
        cv::Moments mnts = cv::moments(wrenchs_contour[i]);
        double match = cv::matchShapes(
            templ_wrench_contour[0], 
            wrenchs_contour[i], 
            cv::CONTOURS_MATCH_I2, 
            0
        );
    
        if (match > 0.9) {  
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

    cv::imshow("res", res);
    cv::imshow("src", img);
    cv::imshow("templ", templ);
    cv::waitKey();
}


int main()
{
    cv::Mat img = cv::imread("/home/vboxuser/Desktop/cv_labs/lab_3/img_zadan/gk/gk.jpg");
    cv::Mat tmpl = cv::imread("/home/vboxuser/Desktop/cv_labs/lab_3/img_zadan/gk/gk_tmplt.jpg");

    gk(img, tmpl);
}