#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>


cv::Point2f contour_center(std::vector<cv::Point> contour)
{
    cv::Moments m = cv::moments(contour);
    return cv::Point2f(m.m10/m.m00, m.m01/m.m00);
}


double contours_ssd(
    std::vector<cv::Point> nomer_contour, 
    std::vector<cv::Point> letter_contour, 
    cv::Point2f nomer_contour_center, 
    cv::Point2f letter_contour_center
){
    double sum = 0;
    for (cv::Point2f p : letter_contour){
        cv::Point2f ps = p - letter_contour_center + nomer_contour_center;
        double min_dist = 1e10;
        for (cv::Point2f q : nomer_contour){
            cv::Point2f delta = cv::Point2f(ps.x - q.x, ps.y - q.y);
            double dist = sqrt(delta.x * delta.x + delta.y * delta.y);
            if (dist < min_dist){
                min_dist = dist;
            }
        }
        sum += min_dist;
    }
    return sum;
}


int main(){
    cv::Mat nomer = cv::imread("/home/vboxuser/Desktop/cv_labs/lab_4/nomer.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat letter = cv::imread("/home/vboxuser/Desktop/cv_labs/lab_4/o.png", cv::IMREAD_GRAYSCALE);

    cv::Mat nomer_bin, letter_bin;
    cv::threshold(nomer, nomer_bin, 150, 255, cv::THRESH_BINARY_INV);
    cv::threshold(letter, letter_bin, 150, 255, cv::THRESH_BINARY_INV);

    std::vector<std::vector<cv::Point>> nomer_contours, letter_contours;
    cv::findContours(nomer_bin, nomer_contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    cv::findContours(letter_bin, letter_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    cv::Point2f letter_contour_center = contour_center(letter_contours[0]);
    for (int i = 0; i < nomer_contours.size(); i++){
        cv::Point2f nomer_contour_center = contour_center(nomer_contours[i]);
        double ssd = contours_ssd(
            nomer_contours[i], 
            letter_contours[0], 
            nomer_contour_center, 
            letter_contour_center
        );
        std::cout << "Contour " << i << " SSD = " << ssd << std::endl;

        cv::Mat view = cv::Mat::zeros(nomer.size(), CV_8UC3);
        cv::drawContours(view, nomer_contours, -1, cv::Scalar(255,255,255));

        std::vector<cv::Point> aligned_letter;
        for(cv::Point2f p : letter_contours[0])
            aligned_letter.push_back(p - letter_contour_center + nomer_contour_center);

        cv::drawContours(view, std::vector<std::vector<cv::Point>>{aligned_letter}, -1, cv::Scalar(0,255,0));
        cv::drawContours(view, nomer_contours, i, cv::Scalar(0,0,255));

        cv::imshow("contour_ssd_aligned", view);
        cv::waitKey();
    }
}
