#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

int main() 
{
    cv::Mat image1 = cv::imread("250px-Lenna.png");
    cv::Mat image2 = cv::imread("Untitled.jpeg");
    double alpha = 0.7;
    
    if (image1.size() != image2.size()){
        cv::resize(image2, image2, image1.size());
    }

    cv::Mat output;

    cv::addWeighted(image1, alpha, image2, 1.0 - alpha, 0.0, output);

    cv::imshow("First image", image1);
    cv::imshow("Second image", image2);
    cv::imshow("Result image", output);

    cv::waitKey();
    cv::destroyAllWindows();
}