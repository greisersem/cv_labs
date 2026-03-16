#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>

using namespace cv;
using namespace std;


static bool readDetectorParameters(string filename, Ptr<aruco::DetectorParameters> &params) {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    
    fs["adaptiveThreshWinSizeMin"] >> params->adaptiveThreshWinSizeMin;
    fs["adaptiveThreshWinSizeMax"] >> params->adaptiveThreshWinSizeMax;
    fs["adaptiveThreshWinSizeStep"] >> params->adaptiveThreshWinSizeStep;
    fs["adaptiveThreshConstant"] >> params->adaptiveThreshConstant;
    fs["minMarkerPerimeterRate"] >> params->minMarkerPerimeterRate;
    fs["maxMarkerPerimeterRate"] >> params->maxMarkerPerimeterRate;
    fs["polygonalApproxAccuracyRate"] >> params->polygonalApproxAccuracyRate;
    fs["minCornerDistanceRate"] >> params->minCornerDistanceRate;
    fs["minDistanceToBorder"] >> params->minDistanceToBorder;
    fs["minMarkerDistanceRate"] >> params->minMarkerDistanceRate;
    fs["cornerRefinementWinSize"] >> params->cornerRefinementWinSize;
    fs["cornerRefinementMaxIterations"] >> params->cornerRefinementMaxIterations;
    fs["cornerRefinementMinAccuracy"] >> params->cornerRefinementMinAccuracy;
    fs["markerBorderBits"] >> params->markerBorderBits;
    fs["perspectiveRemovePixelPerCell"] >> params->perspectiveRemovePixelPerCell;
    fs["perspectiveRemoveIgnoredMarginPerCell"] >> params->perspectiveRemoveIgnoredMarginPerCell;
    fs["maxErroneousBitsInBorderRate"] >> params->maxErroneousBitsInBorderRate;
    fs["minOtsuStdDev"] >> params->minOtsuStdDev;
    fs["errorCorrectionRate"] >> params->errorCorrectionRate;
    return true;
}

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) return -1;

    Ptr<aruco::Dictionary> dict = makePtr<aruco::Dictionary>(
        aruco::getPredefinedDictionary(aruco::DICT_6X6_250)
    );

    Ptr<aruco::DetectorParameters> params = makePtr<aruco::DetectorParameters>();
    readDetectorParameters("/home/greisersem/Desktop/cv_labs/lab_5/detector_params.yml", params);

    Mat camMatrix, distCoeffs;
    FileStorage fs("/home/greisersem/Desktop/cv_labs/lab_5/cam.yml", FileStorage::READ);
    if(!fs.isOpened()){
        cout << "Error: cam.yml not found!" << endl;
        return -1;
    }
    fs["camera_matrix"] >> camMatrix;
    fs["distortion_coefficients"] >> distCoeffs;

    float markerLength = 0.013; 

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
            aruco::estimatePoseSingleMarkers(corners, markerLength, camMatrix, distCoeffs, rvecs, tvecs);

            for(size_t i = 0; i < ids.size(); i++){
                float h = markerLength;
                float half = markerLength / 2.0f;

                vector<Point3f> cubePoints = {
                    {-half, -half, 0}, { half, -half, 0}, { half,  half, 0}, {-half,  half, 0},
                    {-half, -half, h}, { half, -half, h}, { half,  half, h}, {-half,  half, h}
                };

                vector<Point2f> imgPoints;
                projectPoints(cubePoints, rvecs[i], tvecs[i], camMatrix, distCoeffs, imgPoints);

                for(int j = 0; j < 4; j++) {
                    line(frame, imgPoints[j], imgPoints[(j+1)%4], Scalar(255,0,0), 2);
                    line(frame, imgPoints[j+4], imgPoints[((j+1)%4)+4], Scalar(0,255,0), 2);
                    line(frame, imgPoints[j], imgPoints[j+4], Scalar(0,0,255), 2);
                }
            }
        }

        imshow("Cubes", frame);
        if(waitKey(1) == 27) break;
    }
    return 0;
}