#include <opencv2/opencv.hpp>
#include <iostream>


bool A(const cv::Mat& img, int r, int c)
{
    int count = 0;
    uchar p2 = img.at<uchar>(r - 1, c);
    uchar p3 = img.at<uchar>(r - 1, c+1);
    uchar p4 = img.at<uchar>(r, c + 1);
    uchar p5 = img.at<uchar>(r + 1, c + 1);
    uchar p6 = img.at<uchar>(r + 1, c);
    uchar p7 = img.at<uchar>(r + 1, c - 1);
    uchar p8 = img.at<uchar>(r, c - 1);
    uchar p9 = img.at<uchar>(r - 1, c - 1);

    if (p2 == 0 && p3 == 255) count++;
    if (p3 == 0 && p4 == 255) count++;
    if (p4 == 0 && p5 == 255) count++;
    if (p5 == 0 && p6 == 255) count++;
    if (p6 == 0 && p7 == 255) count++;
    if (p7 == 0 && p8 == 255) count++;
    if (p8 == 0 && p9 == 255) count++;
    if (p9 == 0 && p2 == 255) count++;

    return count == 1;
}


bool B(const cv::Mat& img, int r, int c)
{
    int count = 0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (img.at<uchar>(r + i, c + j) == 255)
                count++;
        }
    }
    return (count - 1 >= 2 && count - 1 <= 6);
}


cv::Mat step_one(cv::Mat img)
{
    cv::Mat pix_to_delete = cv::Mat::zeros(img.size(), CV_8UC1);
    for (int row = 1; row < img.rows - 1; row++) {
        for (int col = 1; col < img.cols - 1; col++) {
            if (img.at<uchar>(row, col) != 255) continue;
            
            if (!A(img, row, col)) continue;
            if (!B(img, row, col)) continue;
            
            if (img.at<uchar>(row-1, col) != 0 && 
                img.at<uchar>(row, col+1) != 0 && 
                img.at<uchar>(row+1, col) != 0) continue;
                
            if (img.at<uchar>(row, col+1) != 0 && 
                img.at<uchar>(row+1, col) != 0 && 
                img.at<uchar>(row, col-1) != 0) continue;

            pix_to_delete.at<uchar>(row, col) = 255;
        }
    }
    return pix_to_delete;
}


cv::Mat step_two(cv::Mat img)
{
    cv::Mat pix_to_delete = cv::Mat::zeros(img.size(), CV_8UC1);
    for (int row = 1; row < img.rows - 1; row++) {
        for (int col = 1; col < img.cols - 1; col++) {
            if (img.at<uchar>(row, col) != 255) continue;
            
            if (!A(img, row, col)) continue;
            if (!B(img, row, col)) continue;
            
            if (img.at<uchar>(row - 1, col) != 0 && // P2
                img.at<uchar>(row, col + 1) != 0 && // P4
                img.at<uchar>(row, col - 1) != 0)   // P8
                continue;
                
            if (img.at<uchar>(row - 1, col) != 0 && // P2
                img.at<uchar>(row + 1, col) != 0 && // P6
                img.at<uchar>(row, col - 1) != 0)   // P8
                continue;

            pix_to_delete.at<uchar>(row, col) = 255;
        }
    }

    return pix_to_delete;
}


void zong_sung(cv::Mat img, cv::Mat &thin_img)
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

    thin_img = image_to_process.clone();
}


int main()
{
    cv::VideoCapture cap("/home/greisersem/Desktop/cv_labs/lab_7/Video/1.avi");
    
    // cv::Mat src = cv::imread("/home/greisersem/Desktop/cv_labs/lab_7/Img/1.jpg", cv::IMREAD_GRAYSCALE);
    while (true) {
        cv::Mat frame;
        cap >> frame;
        
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::blur(gray, gray, cv::Size(3, 3));
        cv::Mat binary;
        cv::threshold(gray, binary, 65, 255, cv::THRESH_BINARY);
        cv::Rect crop(0, 0, binary.cols, (int)(binary.rows / 1.8));
        cv::rectangle(binary, crop, cv::Scalar(0, 0, 0), -1);
        cv::erode(binary, binary, cv::Mat());

        cv::Mat tiny;
        cv::resize(binary, tiny, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
    
        cv::Mat result;
        zong_sung(binary, result);

        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(result, lines, 1, CV_PI / 180, 55, 50, 100);

        for (size_t i = 0; i < lines.size(); i++) {
            cv::Vec4i l = lines[i];
            cv::line(frame, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255, 0, 0), 3, cv::LINE_AA);
        }


        cv::imshow("Source", frame);
        cv::imshow("Binary", binary);
        cv::imshow("Result", result);
        
        if (cv::waitKey(1) == 27) break;
    }
    cv::destroyAllWindows();

    cv::Mat coins = cv::imread("/home/greisersem/Desktop/cv_labs/lab_7/Img/coins.jpg");

    cv::Mat gray, blurred;
    cv::cvtColor(coins, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(9, 9), 2, 2);

    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(blurred, circles, cv::HOUGH_GRADIENT, 1, 
                     gray.rows / 8, 
                     100, 30,       
                     30, 100);       

    cv::Mat hsv;
    cv::cvtColor(coins, hsv, cv::COLOR_BGR2HSV);

    for (size_t i = 0; i < circles.size(); i++) {
        cv::Point center(std::round(circles[i][0]), std::round(circles[i][1]));
        int radius = std::round(circles[i][2]);

        cv::Mat mask = cv::Mat::zeros(coins.size(), CV_8UC1);
        cv::circle(mask, center, radius * 0.8, cv::Scalar(255), -1);

        cv::Scalar avg_hsv = cv::mean(hsv, mask);
        double hue = avg_hsv[0];
        double saturation = avg_hsv[1];
        std::cout << i << " " << hue << " " << saturation << std::endl;

        std::string material;
        cv::Scalar label_color;

        if (saturation < 130) {
            material = "Nickel";
            label_color = cv::Scalar(255, 0, 0);
        } else {
            material = "Brass";
            label_color = cv::Scalar(0, 255, 255);
        }

        cv::circle(coins, center, radius, label_color, 3);
    }

    cv::imshow("Coins", coins);
    cv::waitKey(0);
    return 0;
}