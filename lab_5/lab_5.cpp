#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Camera error\n";
        return -1;
    }

    Ptr<aruco::Dictionary> dict = makePtr<aruco::Dictionary>(
        aruco::getPredefinedDictionary(aruco::DICT_6X6_250)
    );

    Ptr<aruco::DetectorParameters> params = makePtr<aruco::DetectorParameters>();

    Mat cameraMatrix, distCoeffs;
    FileStorage fs("/home/greisersem/Desktop/cv_labs/lab_5/cam.yml", FileStorage::READ);
    if(!fs.isOpened()){
        cout << "Error: camera.yml not found!" << endl;
        return -1;
    }
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;

    float markerLength = 0.013; // 13 мм

    while(true){
        Mat frame;
        cap >> frame;
        if(frame.empty()) break;

        vector<int> ids;
        vector<vector<Point2f>> corners;

        aruco::detectMarkers(frame, dict, corners, ids, params);

        if(!ids.empty()){
            aruco::drawDetectedMarkers(frame, corners, ids);

            vector<Vec3d> rvecs, tvecs;
            aruco::estimatePoseSingleMarkers(
                corners,
                markerLength,
                cameraMatrix,
                distCoeffs,
                rvecs,
                tvecs
            );

            for(size_t i = 0; i < ids.size(); i++){
                float h = markerLength;
				float half = markerLength / 2.0f;

				vector<Point3f> cubePoints = {
					{-half, -half, 0}, 
					{ half, -half, 0}, 
					{ half,  half, 0}, 
					{-half,  half, 0},

					{-half, -half, h}, 
					{ half, -half, h}, 
					{ half,  half, h}, 
					{-half,  half, h}
				};

                vector<Point2f> imgPoints;
                projectPoints(cubePoints, rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imgPoints);

                for(int j=0; j<4; j++){
                    line(frame, imgPoints[j], imgPoints[(j+1)%4], Scalar(255,0,0), 2);
                    line(frame, imgPoints[j+4], imgPoints[((j+1)%4)+4], Scalar(0,255,0), 2);
                    line(frame, imgPoints[j], imgPoints[j+4], Scalar(0,0,255), 2);
                }
            }
        }

        imshow("AR Cube", frame);
        if(waitKey(1) == 27) break;
    }
}