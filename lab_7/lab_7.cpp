#include <opencv2/opencv.hpp>
#include <iostream>


bool A(cv::Mat img, cv::Point current_pix)
{
    int count = 0;
    
    if (img.at<uchar>(current_pix.x, current_pix.y + 1) == 0 &&
        img.at<uchar>(current_pix.x + 1, current_pix.y + 1) == 255) {
            count++;
        }
    
    if (img.at<uchar>(current_pix.x + 1, current_pix.y + 1) == 0 &&
        img.at<uchar>(current_pix.x + 1, current_pix.y) == 255) {
            count++;
        }

    if (img.at<uchar>(current_pix.x + 1, current_pix.y) == 0 &&
        img.at<uchar>(current_pix.x + 1, current_pix.y - 1) == 255) {
            count++;
        }

    if (img.at<uchar>(current_pix.x + 1, current_pix.y - 1) == 0 &&
        img.at<uchar>(current_pix.x, current_pix.y - 1) == 255) {
            count++;
        }

    if (img.at<uchar>(current_pix.x, current_pix.y - 1) == 0 &&
        img.at<uchar>(current_pix.x - 1, current_pix.y - 1) == 255) {
            count++;
        }
    
    if (img.at<uchar>(current_pix.x - 1, current_pix.y - 1) == 0 &&
        img.at<uchar>(current_pix.x - 1, current_pix.y) == 255) {
            count++;
        }

    if (img.at<uchar>(current_pix.x - 1, current_pix.y) == 0 &&
        img.at<uchar>(current_pix.x - 1, current_pix.y + 1) == 255) {
            count++;
        }
    
    if (img.at<uchar>(current_pix.x - 1, current_pix.y + 1) == 0 &&
        img.at<uchar>(current_pix.x, current_pix.y + 1) == 255) {
            count++;
        }

    return count == 1;
}


bool B(cv::Mat img, cv::Point current_pix)
{
    int count = 0;
    for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
            if (img.at<uchar>(current_pix.x + j, current_pix.y + i))
                count++;
        }
    }
    count--;
    return (count >= 2 && count <= 6);
}


cv::Mat step_one(cv::Mat img)
{
    cv::Mat pix_to_delete = cv::Mat::zeros(img.size(), CV_8UC1);
    for (int i = 1; i < img.rows - 1; i++) {
        for (int j = 1; j < img.cols - 1; j++) {
            if (img.at<uchar>(j, i) != 255) continue;
            if (!A(img, cv::Point(j, i))) continue;
            if (!B(img, cv::Point(j, i))) continue;
            if (img.at<uchar>(j, i + 1) * 
                img.at<uchar>(j + 1, i) * 
                img.at<uchar>(j, i - 1) != 0
            ) continue;
            if (img.at<uchar>(j - 1, i) * 
                img.at<uchar>(j + 1, i) *
                img.at<uchar>(j, i - 1) != 0
            ) continue;

            pix_to_delete.at<uchar>(j, i) = 255;
        }
    }

    return pix_to_delete;
}


cv::Mat step_two(cv::Mat img)
{
    cv::Mat pix_to_delete = cv::Mat::zeros(img.size(), CV_8UC1);
    for (int i = 1; i < img.rows - 1; i++) {
        for (int j = 1; j < img.cols - 1; j++) {
            if (img.at<uchar>(j, i) != 255) continue;
            if (!A(img, cv::Point(j, i))) continue;
            if (!B(img, cv::Point(j, i))) continue;
            if (img.at<uchar>(j, i + 1) * 
                img.at<uchar>(j + 1, i) *
                img.at<uchar>(j - 1, i) != 0
            ) continue;
            if (img.at<uchar>(j - 1, i) * 
                img.at<uchar>(j, i + 1) *
                img.at<uchar>(j, i - 1) != 0
            ) continue;

            pix_to_delete.at<uchar>(j, i) = 255;
        }
    }

    return pix_to_delete;
}


void zong_sung(cv::Mat img, cv::Mat thin_img)
{
    cv::Mat image_to_process = img.clone();
    
    cv::Mat diff;
    cv::Mat prev;

    do {
        image_to_process.copyTo(prev);

        cv::Mat mask_1 = step_one(image_to_process);
        image_to_process -= mask_1;

        cv::Mat mask_2 = step_two(image_to_process);
        image_to_process -= mask_2;

        cv::absdiff(image_to_process, prev, diff);
    } while (cv::countNonZero(diff) != 0);
}


int main()
{
    cv::Mat src = cv::imread("input.png", cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "Error: Image not found!" << std::endl;
        return -1;
    }

    cv::Mat binary;
    cv::threshold(src, binary, 127, 255, cv::THRESH_BINARY);

    cv::Mat result;
    zong_sung(binary, result);

    cv::imshow("Source", binary);
    cv::imshow("Result", result);
    cv::waitKey(0);
}