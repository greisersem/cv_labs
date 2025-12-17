#include <cmath>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


void DFT(const cv::Mat& src, cv::Mat& dst)
{
    int M = src.rows;
    int N = src.cols;

    dst = cv::Mat::zeros(M, N, CV_32FC2);

    for (int u = 0; u < M; ++u)
    {
        std::cout << u << "lane" << std::endl;
        float angle_u = -2.0 * CV_PI * u / M;
        std::complex<float> Wu(cos(angle_u), sin(angle_u));
        std::complex<float> Wu_pow(1.0, 0.0);

        for (int v = 0; v < N; ++v)
        {
            float angle_v = -2.0 * CV_PI * v / N;
            std::complex<float> Wv(cos(angle_v), sin(angle_v));
            std::complex<float> sum(0.0, 0.0);
            std::complex<float> Wv_pow(1.0, 0.0);

            for (int x = 0; x < M; ++x)
            {
                for (int y = 0; y < N; ++y)
                {
                    float pixel = src.at<float>(x, y);

                    sum += pixel * Wv_pow;
                    Wv_pow *= Wv;
                }

            }

            std::complex<float> result = sum * Wu_pow;
            dst.at<cv::Vec2f>(u, v)[0] += result.real();
            dst.at<cv::Vec2f>(u, v)[1] += result.imag();
        }
        Wu_pow *= Wu;
    }
}


void IDFT(const cv::Mat& src, cv::Mat& dst)
{
    int M = src.rows;
    int N = src.cols;

    dst = cv::Mat::zeros(M, N, CV_32F);

    for (int x = 0; x < M; ++x)
    {
        std::cout << x << "lane" << std::endl;
        float angle_x = 2.0 * CV_PI * x / M;
        std::complex<float> Wx(1.0, 0.0);
        std::complex<float> Wx_base(cos(angle_x), sin(angle_x));

        for (int y = 0; y < N; ++y)
        {
            float angle_y = 2.0 * CV_PI * y / N;
            std::complex<float> Wy(1.0, 0.0);
            std::complex<float> Wy_base(cos(angle_y), sin(angle_y));

            std::complex<float> sum(0.0, 0.0);

            for (int u = 0; u < M; ++u)
            {
                for (int v = 0; v < N; ++v)
                {
                    cv::Vec2f Fuv = src.at<cv::Vec2f>(u, v);
                    std::complex<float> c(Fuv[0], Fuv[1]);

                    sum += c * Wx * Wy;
                    Wy *= Wy_base;
                }

                Wx *= Wx_base;

                Wy = std::complex<float>(1.0, 0.0);
            }
            dst.at<float>(x, y) = sum.real() / (M * N);
        }
    }
}


int main()
{
    // Загружаем изображение в grayscale
    cv::Mat img = cv::imread("/home/vboxuser/Desktop/cv_labs/lab_4/fftdemo.jpg", cv::IMREAD_GRAYSCALE);
    if(img.empty()) {
        std::cerr << "Ошибка: не удалось открыть изображение!\n";
        return -1;
    }

    // Переводим в float
    img.convertTo(img, CV_32F);

    // Вычисляем DFT
    cv::Mat dft, idft;
    DFT(img, dft);

    // Получаем модуль спектра для отображения
    cv::Mat planes[2], mag;
    cv::split(dft, planes);                     // split на Re и Im
    cv::magnitude(planes[0], planes[1], mag);  // вычисляем sqrt(Re^2 + Im^2)
    cv::normalize(mag, mag, 0, 1, cv::NORM_MINMAX); // нормализация для отображения
    cv::imshow("DFT Magnitude", mag);

    // Вычисляем обратное преобразование
    IDFT(dft, idft);
    cv::normalize(idft, idft, 0, 1, cv::NORM_MINMAX); // нормализация
    cv::imshow("IDFT", idft);

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}

