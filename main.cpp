#include "vo.h"

using namespace std;

string datasetPath = "/home/prostoichelovek/Documents/datasets/dataset";

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

    VO vo(K, baseline, left_path, right_path);

    //we have only 1000 paris of pictures
    vo.set_Max_frame(max_frame);

    //This is to get the ground truethe pose data
    vector<vector<float>> poses = VO::get_Pose(pose_path);

    vo.playSequence(poses);

    cout << "You come to the end!" << endl;

    return 0;
}
