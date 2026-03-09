#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

const int CAM_ANGLE_X = 74; // deg
const float CAM_ANGLE_X_RAD = CAM_ANGLE_X * CV_PI / 180.0f;
const int Y = -250;          // mm
const float SCALE = 0.2;
float D_LASER;
cv::Vec3f N_LASER;


void get_laser_for_calib(const cv::Mat &frame, cv::Mat &binarized_img)
{
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, gray, 170, 255, cv::THRESH_BINARY);
    cv::blur(gray, gray, cv::Size(3, 3));
    cv::Canny(gray, binarized_img, 110, 150);
}


void get_laser(const cv::Mat &frame, cv::Mat &binarized_img)
{
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(
        hsv,
        cv::Scalar(0, 0, 90),
        cv::Scalar(93, 115, 255),
        binarized_img 
    );
}


cv::Vec3f get_e_vector(cv::Point2f p, float fx, float fy, cv::Point2f center) {
    float x_foc = (center.x - p.x) / fx;
    float y_foc = (center.y - p.y) / fy;

    cv::Vec3f v(x_foc, y_foc, 1.0f);

    return cv::normalize(v);
}


void calibration()
{
    cv::Mat calib_img = cv::imread("/home/vboxuser/Desktop/cv_labs/lab_6/Video/calib_1_0.jpg");
    std::cout << calib_img.cols << "x" << calib_img.rows << std::endl;
    cv::Mat binarized_img;
    std::vector<cv::Vec2f> lines;

    int w = calib_img.cols;
    int h = calib_img.rows;
    cv::Point2f center(w / 2.0f, h / 2.0f);
    float fx = (w / 2.0f) / tan(CAM_ANGLE_X_RAD / 2.0f);
    float fy = fx;

    get_laser_for_calib(calib_img, binarized_img);

    cv::imshow("binarized", binarized_img);
    cv::waitKey();

    cv::HoughLines(binarized_img, lines, 1.0, CV_PI / 90, 37);
    std::cout << "lines found: " << lines.size() << std::endl;

    std::vector<cv::Vec3f> p_3d;
    cv::Mat lines_img = calib_img.clone();
    for (int i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0];
        float theta = lines[i][1];
        
        float Z = 0;
        if (rho > 350) Z = 340.0;     
        else if (rho > 280) Z = 500.0;
        else if (rho > 250) Z = 840.0;
        else if (rho > 230) Z = 1140.0;

        double a = cos(theta), b = sin(theta);
        cv::Point2f p1(0, rho / b); 
        cv::Point2f p2(calib_img.cols, (rho - calib_img.cols * a) / b);
        cv::line(lines_img, p1, p2, cv::Scalar(0, 255, 0), 2);
        cv::Mat img = lines_img.clone();
        cv::putText(img, std::to_string(Z), cv::Point(10, 30), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255), 2);
        cv::putText(img, std::to_string(rho), cv::Point(10, 60), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255), 2);
        cv::imshow("lines", img);
        cv::waitKey();

        cv::Vec3f e1 = get_e_vector(p1, fx, fy, center);
        cv::Vec3f e2 = get_e_vector(p2, fx, fy, center);
        // std::cout << e1 << std::endl;
        // std::cout << e2 << std::endl;

        float k1 = Z / e1[2];
        float k2 = Z / e2[2];
        cv::Vec3f p1_3d = k1 * e1;
        cv::Vec3f p2_3d = k2 * e2;

        p_3d.push_back(p1_3d);
        p_3d.push_back(p2_3d);
    }
    

    std::vector<cv::Vec3f> h_rows;
    cv::Vec3f p_ref = p_3d[0];
    for (int i = 0; i < p_3d.size(); i++) {
        cv::Vec3f h = p_ref - p_3d[i];
        h_rows.push_back(h);
    }


    cv::Mat H(h_rows.size(), 3, CV_32F);
    for (int i = 0; i < h_rows.size(); i++) {
        H.at<float>(i, 0) = h_rows[i][0];
        H.at<float>(i, 1) = h_rows[i][1];
        H.at<float>(i, 2) = h_rows[i][2];
    }

    cv::Mat K = H.t() * H; // covariation matrix

    cv::Mat eigenvalues, eigenvectors;
    cv::eigen(K, eigenvalues, eigenvectors);

    cv::Vec3f n_laser;
    n_laser[0] = eigenvectors.at<float>(2, 0);
    n_laser[1] = eigenvectors.at<float>(2, 1);
    n_laser[2] = eigenvectors.at<float>(2, 2);

    n_laser = cv::normalize(n_laser);

    float d_laser = -(n_laser[0] * p_3d[0][0] + n_laser[1] * p_3d[0][1] + n_laser[2] * p_3d[0][2]);

    std::cout << "--- РЕЗУЛЬТАТЫ КАЛИБРОВКИ ---" << std::endl;
    std::cout << "Нормаль n: " << n_laser << std::endl;
    std::cout << "Коэффициент d: " << d_laser << std::endl;

    D_LASER = d_laser;
    N_LASER = n_laser;

    cv::destroyAllWindows();
}


int main()
{
    cv::VideoCapture cap("/home/vboxuser/Desktop/cv_labs/lab_6/Video/1.avi");

    int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    cv::Point2i center(width / 2, height / 2);
    std::cout << width << "x" << height << std::endl;

    const float fx = (width / 2.0f) / tan(CAM_ANGLE_X_RAD / 2.0f);
    const float fy = fx;

    calibration();

    cv::Mat frame, binarized_img;

    while (cap.read(frame)) {
        cv::Mat map = cv::Mat::zeros(height, width, CV_8UC3);

        for (int i = 0; i < width; i += 100 * SCALE) {
            cv::line(map, cv::Point(i, 0), cv::Point(i, height), cv::Scalar(255, 255, 255));
            int x_cm = static_cast<int>((i - width / 2) / (SCALE * 10));
            cv::putText(map, std::to_string(x_cm), cv::Point(i + 1, 20), 
                cv::FONT_HERSHEY_PLAIN, 0.5, cv::Scalar(255, 255, 255));
            if (i < height) {
                cv::line(map, cv::Point(0, i), cv::Point(width, i), cv::Scalar(255, 255, 255));
                int z_cm = static_cast<int>(i / (SCALE * 10));
                cv::putText(map, std::to_string(z_cm), cv::Point(width - 20, i - 1), 
                            cv::FONT_HERSHEY_PLAIN, 0.5, cv::Scalar(255, 255, 255));
            }
        }

        get_laser(frame, binarized_img);

        for (int pix_y = 0; pix_y < height; pix_y++) {
            for (int pix_x = 0; pix_x < width; pix_x++) {
                if (binarized_img.at<uchar>(pix_y, pix_x) == 255) {
                    cv::Point p(pix_x, pix_y);
                    cv::Vec3f e = get_e_vector(p, fx, fy, center);

                    float denominator = N_LASER[0] * e[0] + N_LASER[1] * e[1] + N_LASER[2] * e[2];
                    if (std::abs(denominator) < 1e-6) continue;

                    double k = - D_LASER / denominator;
                    cv::Vec3f p_real = e * k;
                    double x_real = p_real[0];
                    double z_real = p_real[2];

                    int map_x = static_cast<int>(x_real * SCALE) + width / 2;
                    int map_y = static_cast<int>(z_real * SCALE);

                    if (map_x >= 0 && map_x < width && map_y >= 0 && map_y < height)
                        cv::circle(map, cv::Point(map_x, map_y), 1, cv::Scalar(0, 255, 0), -1);
                }
            }
        }

        cv::imshow("Video", frame);
        cv::imshow("Laser mask", binarized_img);
        cv::imshow("Map", map);

        if (cv::waitKey(30) == 27) {
            break;
        }
    }

    return 0;
}