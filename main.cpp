#include <opencv2/opencv.hpp>

#include "vo.h"

using namespace std;
using namespace cv;

string datasetPath = "/home/prostoichelovek/Documents/datasets/dataset";

Mat getImage(const string &raw_path, int i) {
    char path[200];
    sprintf(path, raw_path.c_str(), i);
    Mat img = imread(path);
    if (!img.data) {
        cerr << "Could not open or find the image" << std::endl;
    }
    return img;
}

int main(int argc, char **argv) {
    int max_frame = 600;
    if (argc > 1)
        max_frame = atoi(argv[1]);

    //you have to configure your own path
    string left_path = datasetPath + "/sequences/00/image_0/%06d.png";
    string right_path = datasetPath + "/sequences/00/image_1/%06d.png";
    string pose_path = datasetPath + "/poses/00.txt";

    cout << "Program starts!" << endl;

    cv::Mat K = (cv::Mat_<double>(3, 3) << 7.188560000000e+02, 0, 6.071928000000e+02,
            0, 7.188560000000e+02, 1.852157000000e+02,
            0, 0, 1);

    float baseline = 0.54;

    VO vo(K, baseline);

    //This is to get the ground truethe pose data
    vector<vector<float>> poses = VO::get_Pose(pose_path);

    Mat traj = Mat::zeros(600, 600, CV_8UC3);

    Mat leftImg = getImage(left_path, 0);
    Mat rightImg = getImage(right_path, 0);
    if (!leftImg.data || !rightImg.data)
        return EXIT_FAILURE;
    vo.prevImg = leftImg;

    vector<Point3f> landmarks;
    vector<Point2f> featurePoints;
    vo.extract_keypoints_surf(leftImg, rightImg, landmarks, featurePoints);

    for (int i = 1; i < max_frame; i++) {
        cout << i << endl;

        leftImg = getImage(left_path, i);
        rightImg = getImage(right_path, i);
        if (!leftImg.data || !rightImg.data)
            continue;

        shared_ptr<Point2f> pos = vo.computePose(leftImg, rightImg, landmarks, featurePoints);

        if (pos != nullptr) {
            rectangle(traj, Point2f(10, 30), Point2f(550, 50), Scalar(0, 0, 0), cv::FILLED);
            putText(traj, "estimated", Point2f(10, 50), cv::FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255), 1, 5);
            circle(traj, Point2f(pos->x + 300, pos->y + 100), 1, Scalar(0, 0, 255), 2);
        }
        putText(traj, "groundtruth", Point2f(10, 70), cv::FONT_HERSHEY_PLAIN, 1, Scalar(255, 0, 0), 1, 5);
        circle(traj, Point(poses[i][3] + 300, poses[i][11] + 100), 1, Scalar(255, 0, 0), 2);

        imshow("Trajectory", traj);
        imshow("image", leftImg);
        char key = waitKey(1);
        if (key == 27)
            break;
    }

    cout << "You come to the end!" << endl;

    return EXIT_SUCCESS;
}
