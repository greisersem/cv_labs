#include <iostream>
#include <vector>
#include <complex>
#include <opencv2/opencv.hpp>


void get_magnitude_spectrum(const cv::Mat& complexImg, cv::Mat& spectrum)
{
    cv::Mat planes[2];
    cv::split(complexImg, planes);

    cv::magnitude(planes[0], planes[1], spectrum);

    spectrum += cv::Scalar::all(1);
    cv::log(spectrum, spectrum);

    cv::normalize(
        spectrum,
        spectrum,
        0,
        1,
        cv::NORM_MINMAX
    );
}


void krasivSpektr(cv::Mat &magI)
{
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;

	cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));  // Top-Right
	cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));  // Bottom-Left
	cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

	cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);
}


void DFT(const cv::Mat& src, cv::Mat& dst)
{
    int M = src.rows;
    int N = src.cols;

    dst = cv::Mat::zeros(M, N, CV_32FC2);

    for (int u = 0; u < M; ++u)
    {
        // std::cout << u << " lane" << std::endl;

        float angle_u = -2.0f * CV_PI * u / M;
        std::complex<float> Wu_base(cos(angle_u), sin(angle_u));

        for (int v = 0; v < N; ++v)
        {
            float angle_v = -2.0f * CV_PI * v / N;
            std::complex<float> Wv_base(cos(angle_v), sin(angle_v));

            std::complex<float> sum(0.0f, 0.0f);

            std::complex<float> Wu_pow(1.0f, 0.0f);

            for (int x = 0; x < M; ++x)
            {
                std::complex<float> Wv_pow(1.0f, 0.0f);

                for (int y = 0; y < N; ++y)
                {
                    float pixel = src.at<float>(x, y);
                    sum += pixel * Wu_pow * Wv_pow;

                    Wv_pow *= Wv_base;
                }

                Wu_pow *= Wu_base;
            }

            dst.at<cv::Vec2f>(u, v)[0] = sum.real();
            dst.at<cv::Vec2f>(u, v)[1] = sum.imag();
        }
    }
}

void IDFT(const cv::Mat& src, cv::Mat& dst)
{
    int M = src.rows;
    int N = src.cols;

    dst = cv::Mat::zeros(M, N, CV_32F);

    for (int x = 0; x < M; ++x)
    {
        // std::cout << x << " lane idft" << std::endl;

        float angle_x = 2.0f * CV_PI * x / M;
        std::complex<float> Wx_base(cos(angle_x), sin(angle_x));

        for (int y = 0; y < N; ++y)
        {
            float angle_y = 2.0f * CV_PI * y / N;
            std::complex<float> Wy_base(cos(angle_y), sin(angle_y));

            std::complex<float> sum(0.0f, 0.0f);

            std::complex<float> Wx_pow(1.0f, 0.0f);

            for (int u = 0; u < M; ++u)
            {
                std::complex<float> Wy_pow(1.0f, 0.0f);

                for (int v = 0; v < N; ++v)
                {
                    cv::Vec2f Fuv = src.at<cv::Vec2f>(u, v);
                    std::complex<float> F(Fuv[0], Fuv[1]);

                    sum += F * Wx_pow * Wy_pow;

                    Wy_pow *= Wy_base;
                }

                Wx_pow *= Wx_base;
            }

            dst.at<float>(x, y) = sum.real() / (M * N);
        }
    }
}


void radix(std::vector<std::complex<float>>& src, std::vector<std::complex<float>>& dst) 
{
    const int n = src.size();
  
    if ((n & (n - 1)) != 0) {
        std::cout << "error in radix size: " << (n & (n - 1));
        return;
    }

    if (n == 1) {
        dst = src;
        return;
    }

    std::vector<std::complex<float>> src0(n / 2);
    std::vector<std::complex<float>> src1(n / 2);

    for (int i = 0; i < n / 2; i++) {
        src0[i] = src[2 * i];
        src1[i] = src[2 * i + 1];
    }

    std::vector<std::complex<float>> src_0(n / 2);
    std::vector<std::complex<float>> src_1(n / 2);

    radix(src0, src_0);
    radix(src1, src_1);

    dst.resize(n);
    double angle = 2 * M_PI / n;
    std::complex<float> w(1);
    std::complex<float> wn(cos(angle), sin(angle));

    for (int i = 0; i < n / 2; i++) {
        dst[i] = src_0[i] + w * src_1[i];
        dst[i + n / 2] = src_0[i] - w * src_1[i];
        w *= wn;
    }
}


std::vector<std::complex<float>> convert_mat_to_vector(cv::Mat img)
{
    std::vector<float> image_vector(img.begin<float>(), img.end<float>());
    std::vector<std::complex<float>> vector;

    for (const auto &val : image_vector)
    {
        vector.push_back(std::complex<float>(val, 0));
    }
    return vector;
}


cv::Mat convert_vector_to_mat(std::vector<std::complex<float>> src, int N) {
	int rows = N;
	int cols = N;
	cv::Mat res(rows, cols, CV_32FC2);

	int k = 0;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			res.at<cv::Vec2f>(i, j)[1] = src[k].imag();
			res.at<cv::Vec2f>(i, j)[0] = src[k].real();
			k++;
		}
	}

	cv::rotate(res, res, cv::ROTATE_90_COUNTERCLOCKWISE);
	return res;
}


int main()
{
    cv::Mat img = cv::imread("/home/vboxuser/Desktop/cv_labs/lab_4/radix.png", cv::IMREAD_GRAYSCALE);

    cv::resize(img, img, cv::Size(128, 128), 0, 0, cv::INTER_AREA);

    img.convertTo(img, CV_32F, 1.0 / 255.0);

    cv::Mat dft, dft_spectrum, idft;

    double tick_start = cv::getTickCount();
    // DFT(img, dft);
    double time_end = (double)(cv::getTickCount() - tick_start) / cv::getTickFrequency();
    // std::cout << "Custom DFT time: " << time_end << std::endl;

    // get_magnitude_spectrum(dft, dft_spectrum);
    // krasivSpektr(dft_spectrum);

    // IDFT(dft, idft);
    // cv::normalize(idft, idft, 0, 1, cv::NORM_MINMAX);

    cv::Mat temp = img.clone();
    int N = img.rows;
    std::vector<std::complex<float>> radix_vect = convert_mat_to_vector(temp);
    std::vector<std::complex<float>> radix_result;

    tick_start = cv::getTickCount();
    radix(radix_vect, radix_result);
    time_end = (double)(cv::getTickCount() - tick_start) / cv::getTickFrequency();
    std::cout << "Radix-2 time: " << time_end << std::endl;

    cv::Mat radix_mat = convert_vector_to_mat(radix_result, N);

    cv::Mat radix_spectrum;
    get_magnitude_spectrum(radix_mat, radix_spectrum);
    krasivSpektr(radix_spectrum);
    cv::flip(radix_spectrum, radix_spectrum, 1);

    std::vector<std::complex<float>> radix_idft_vect;
    radix(radix_result, radix_idft_vect);
    for (auto& v : radix_idft_vect) {
        v /= radix_idft_vect.size();
    }

    cv::Mat radix_idft(img.rows, img.cols, CV_32F);

    int k = 0;
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            radix_idft.at<float>(i, j) =
                radix_idft_vect[k].real();
            k++;
        }
    }

    cv::normalize(radix_idft, radix_idft, 0, 1, cv::NORM_MINMAX);
    cv::flip(radix_idft, radix_idft, 0);
    cv::flip(radix_idft, radix_idft, 1);
    
    cv::Mat dft_opencv, dft_opencv_spectrum, idft_opencv;
    tick_start = cv::getTickCount();
    cv::dft(img, dft_opencv, cv::DFT_COMPLEX_OUTPUT);
    time_end = (double)(cv::getTickCount() - tick_start) / cv::getTickFrequency();
    std::cout << "DFT OpenCV time: " << time_end << std::endl;

    get_magnitude_spectrum(dft_opencv, dft_opencv_spectrum);
    krasivSpektr(dft_opencv_spectrum);
    cv::dft(dft_opencv, idft_opencv, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
    
    cv::imshow("Original", img);
    // cv::imshow("DFT Magnitude", dft_spectrum);
    // cv::imshow("IDFT", idft);
    cv::imshow("Radix-2 Magnitude", radix_spectrum);
    cv::imshow("Radix idft", radix_idft);
    cv::imshow("DFT OpenCV", dft_opencv_spectrum);
    cv::imshow("IDFT OpenCV", idft_opencv);


    cv::waitKey();
}

