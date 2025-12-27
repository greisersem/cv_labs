#include <iostream>
#include <vector>
#include <complex>
#include <opencv2/opencv.hpp>
#include <opencv2/intensity_transform.hpp>


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

void dft_filter(const cv::Mat& img, const cv::Mat& kernel, std::string window_name)
{
    CV_Assert(img.type() == CV_32F);
    CV_Assert(kernel.type() == CV_32F);

    int P = img.rows + kernel.rows - 1;
    int Q = img.cols + kernel.cols - 1;

    cv::Mat padded_img, padded_kernel;
    cv::copyMakeBorder(img, padded_img,
        0, P - img.rows, 0, Q - img.cols,
        cv::BORDER_CONSTANT, 0);

    cv::copyMakeBorder(kernel, padded_kernel,
        0, P - kernel.rows, 0, Q - kernel.cols,
        cv::BORDER_CONSTANT, 0);

    cv::Mat F, H;
    cv::dft(padded_img, F, cv::DFT_COMPLEX_OUTPUT);
    cv::dft(padded_kernel, H, cv::DFT_COMPLEX_OUTPUT);
    
    cv::Mat G, G_spectrum;
    cv::mulSpectrums(F, H, G, 0);

    get_magnitude_spectrum(G, G_spectrum);
    krasivSpektr(G_spectrum);
    cv::imshow(window_name, G_spectrum);

    cv::Mat conv;
    cv::dft(G, conv, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

    cv::Mat result = conv(cv::Rect(0, 0, img.cols, img.rows)).clone();
    cv::normalize(result, result, 0, 1, cv::NORM_MINMAX);
    cv::imshow(window_name + "idft", result);
}


cv::Mat low_filter(cv::Mat src, int rad) {
    cv::Mat padded;
    int m = cv::getOptimalDFTSize(src.rows);
    int n = cv::getOptimalDFTSize(src.cols);
    cv::copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
    cv::Mat complex;
    cv::merge(planes, 2, complex);
    cv::dft(complex, complex);
    krasivSpektr(complex);
    cv::Mat mask = cv::Mat::zeros(complex.rows, complex.cols, CV_32F);
    cv::circle(mask, cv::Point(mask.cols / 2, mask.rows / 2), rad, cv::Scalar(1, 1, 1), -1, 8, 0);
    cv::Mat filtered;
    complex.copyTo(filtered);
    cv::split(filtered, planes);
    planes[0] = planes[0].mul(mask);
    planes[1] = planes[1].mul(mask);
    cv::merge(planes, 2, filtered);
    cv::split(filtered, planes);
    cv::magnitude(planes[0], planes[1], planes[0]);
    cv::Mat mag = planes[0];
    mag += cv::Scalar::all(1);
    cv::log(mag, mag);
    cv::normalize(mag, mag, 0, 1, cv::NORM_MINMAX);
    krasivSpektr(filtered);
    cv::Mat inversed;
    cv::idft(filtered, inversed, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    cv::normalize(inversed, inversed, 0, 1, cv::NORM_MINMAX);
    imshow("lf", mag);
    return inversed;
}


cv::Mat high_filter(cv::Mat src, int rad) {
    cv::Mat padded;
    int m = cv::getOptimalDFTSize(src.rows);
    int n = cv::getOptimalDFTSize(src.cols);
    cv::copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
    cv::Mat complex;
    cv::merge(planes, 2, complex); 
    cv::dft(complex, complex);
    krasivSpektr(complex);
    cv::Mat mask = cv::Mat::ones(complex.rows, complex.cols, CV_32F);
    cv::circle(mask, cv::Point(mask.cols / 2, mask.rows / 2), rad, cv::Scalar(0, 0, 0), -1, 8, 0); 
    cv::Mat filtered;
    complex.copyTo(filtered);
    cv::split(filtered, planes);
    planes[0] = planes[0].mul(mask);
    planes[1] = planes[1].mul(mask); 
    cv::merge(planes, 2, filtered);
    cv::split(filtered, planes); 
    cv::magnitude(planes[0], planes[1], planes[0]); 
    cv::Mat mag = planes[0];
    mag += cv::Scalar::all(1); 
    cv::log(mag, mag);
    cv::normalize(mag, mag, 0, 1, cv::NORM_MINMAX);
    krasivSpektr(filtered);
    cv::Mat inversed;
    cv::idft(filtered, inversed, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT); 
    cv::normalize(inversed, inversed, 0, 1, cv::NORM_MINMAX);
    imshow("hf", mag);
    return inversed;
}


cv::Mat correlation(cv::Mat& src, cv::Mat& templ)
{
    CV_Assert(src.type() == CV_32F && templ.type() == CV_32F);
    cv::Scalar meanSymbol = cv::mean(templ);
    cv::Scalar meanImg = cv::mean(src);

    src = src - meanImg[0];
    templ = templ - meanSymbol[0];
    int P = cv::getOptimalDFTSize(src.rows + templ.rows - 1);
    int Q = cv::getOptimalDFTSize(src.cols + templ.cols - 1);

    cv::Mat padded_src, padded_templ;
    cv::copyMakeBorder(src, padded_src, 0, P - src.rows, 0, Q - src.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::copyMakeBorder(templ, padded_templ, 0, P - templ.rows, 0, Q - templ.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    cv::Mat F, H;
    cv::dft(padded_src, F, cv::DFT_COMPLEX_OUTPUT);
    cv::dft(padded_templ, H, cv::DFT_COMPLEX_OUTPUT);

    cv::Mat G;
    cv::mulSpectrums(F, H, G, 0, true);

    cv::Mat corr, corr_spectrum;
    cv::idft(G, corr, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    cv::Mat corr_valid = corr(cv::Rect(0, 0, src.cols - templ.cols + 1, src.rows - templ.rows + 1)).clone();
    cv::normalize(
        corr,
        corr_spectrum,
        0,
        1,
        cv::NORM_MINMAX
    );
    cv::Mat spectrumROI = corr_spectrum(cv::Rect(0, 0, src.cols, src.rows)).clone();
    cv::imshow("corr", spectrumROI);
    cv::normalize(corr_valid, corr_valid, 0, 1, cv::NORM_MINMAX);

    double maxVal;
    cv::minMaxLoc(corr_valid, nullptr, &maxVal);
    double threshold = maxVal - 0.09;
    cv::Mat binary;
    cv::threshold(corr_valid, binary, threshold, 1.0, cv::THRESH_BINARY);

    return binary;
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
    cv::destroyAllWindows();

    cv::Mat laplas, box, sobel_x, sobel_y;

    cv::Mat laplas_kernel(cv::Size(3, 3), CV_32FC1, cv::Scalar());
	laplas_kernel.at<float>(0, 0) = 0;
	laplas_kernel.at<float>(0, 1) = 1;
	laplas_kernel.at<float>(0, 2) = 0;
	laplas_kernel.at<float>(1, 0) = 1;
	laplas_kernel.at<float>(1, 1) = -4;
	laplas_kernel.at<float>(1, 2) = 1;
	laplas_kernel.at<float>(2, 0) = 0;
	laplas_kernel.at<float>(2, 1) = 1;
	laplas_kernel.at<float>(2, 2) = 0;

    cv::Mat sobel_y_kernel(cv::Size(3, 3), CV_32FC1, cv::Scalar());
    sobel_y_kernel.at<float>(0, 0) = 1;
	sobel_y_kernel.at<float>(0, 1) = 2;
	sobel_y_kernel.at<float>(0, 2) = 1;
	sobel_y_kernel.at<float>(1, 0) = 0;
	sobel_y_kernel.at<float>(1, 1) = 0;
	sobel_y_kernel.at<float>(1, 2) = 0;
	sobel_y_kernel.at<float>(2, 0) = -1;
	sobel_y_kernel.at<float>(2, 1) = -2;
	sobel_y_kernel.at<float>(2, 2) = -1;

    cv::Mat sobel_x_kernel(cv::Size(3, 3), CV_32FC1, cv::Scalar());
    sobel_x_kernel.at<float>(0, 0) = 1;
	sobel_x_kernel.at<float>(0, 1) = 0;
	sobel_x_kernel.at<float>(0, 2) = -1;
	sobel_x_kernel.at<float>(1, 0) = 2;
	sobel_x_kernel.at<float>(1, 1) = 0;
	sobel_x_kernel.at<float>(1, 2) = -2;
	sobel_x_kernel.at<float>(2, 0) = 1;
	sobel_x_kernel.at<float>(2, 1) = 0;
	sobel_x_kernel.at<float>(2, 2) = -1;

    cv::Mat box_kernel(cv::Size(3, 3), CV_32FC1, cv::Scalar());
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			box_kernel.at<float>(i, j) = 1;
		}
	}
    box_kernel /= 9.0f;

    dft_filter(img, laplas_kernel, "Laplas");
    dft_filter(img, sobel_x_kernel, "Sobel X");
    dft_filter(img, sobel_y_kernel, "Sobel Y");
    dft_filter(img, box_kernel, "Box filter");

    cv::waitKey();
    cv::destroyAllWindows();

    cv::Mat high = high_filter(img, 20);
    cv::Mat low = low_filter(img, 20);

    cv::imshow("Low filter", low);
    cv::imshow("High filter", high);
    cv::waitKey();
    cv::destroyAllWindows();

    cv::Mat nomer, letter, letter_1, letter_2;
    nomer = cv::imread("/home/vboxuser/Desktop/cv_labs/lab_4/nomer.jpg", cv::IMREAD_GRAYSCALE);
    letter = cv::imread("/home/vboxuser/Desktop/cv_labs/lab_4/o.png", cv::IMREAD_GRAYSCALE);
    letter_1 = cv::imread("/home/vboxuser/Desktop/cv_labs/lab_4/letter_1.jpg", cv::IMREAD_GRAYSCALE);
    letter_2 = cv::imread("/home/vboxuser/Desktop/cv_labs/lab_4/image.png", cv::IMREAD_GRAYSCALE);

    nomer.convertTo(nomer, CV_32F, 1.0 / 255.0);
    letter.convertTo(letter, CV_32F, 1.0 / 255.0);
    letter_1.convertTo(letter_1, CV_32F, 1.0 / 255.0);
    letter_2.convertTo(letter_2, CV_32F, 1.0 / 255.0);

    cv::Mat result, result_1, result_2;
    cv::imshow("Nomer", nomer);
    result = correlation(nomer, letter);
    cv::imshow("Letter", letter);
    cv::imshow("Correlation", result);
	cv::waitKey();

    result_1 = correlation(nomer, letter_1);
    cv::imshow("Letter 2", letter_1);
    cv::imshow("Correlation_1", result_1);
	cv::waitKey();

    result_2 = correlation(nomer, letter_2);
    cv::imshow("Number", letter_2);
    cv::imshow("Correlation_2", result_2);
	cv::waitKey();
    cv::destroyAllWindows();
}

