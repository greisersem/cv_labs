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
    double templ_perimeter = m00;

    cv::cvtColor(img, res, cv::COLOR_BGR2GRAY);
    cv::threshold(res, res, 230, 255, cv::THRESH_BINARY_INV);
    std::vector <std::vector <cv::Point>> wrenchs_contour;
    cv::findContours(res, wrenchs_contour, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    double match;

    for (int contour_index = 0; contour_index < wrenchs_contour.size(); contour_index++) {
        cv::Moments mnts = cv::moments(wrenchs_contour[contour_index]);
        double temp_area = cv::contourArea(wrenchs_contour[contour_index]);
        double match = std::abs((templ_area / (templ_perimeter + 10)) - (temp_area / (mnts.m00 + 10)));

        if (match > 0.0005 && match < 0.0006) {  
            cv::polylines(img, wrenchs_contour[contour_index], true, cv::Scalar(0, 255, 0), 5, 8);
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
            cv::polylines(img, wrenchs_contour[contour_index], true, cv::Scalar(0, 0, 255), 5, 8);
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