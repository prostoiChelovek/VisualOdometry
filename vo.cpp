#include "vo.h"

using namespace std;
using namespace cv;


VO::VO(const cv::Mat &_K, float _baseline)
        : K(_K), baseline(_baseline) {}

VO::VO(const cv::Mat &_K, float _baseline, const cv::Mat &firstImg)
        : prevImg(firstImg), K(_K), baseline(_baseline) {}

vector<Point3f> VO::get3D_Points(const vector<Point2f> &feature_p1, const vector<Point2f> &feature_p2) const {
    // This is to generate the 3D points

    Mat M_left = Mat::zeros(3, 4, CV_64F);
    Mat M_right = Mat::zeros(3, 4, CV_64F);
    M_left.at<double>(0, 0) = 1;
    M_left.at<double>(1, 1) = 1;
    M_left.at<double>(2, 2) = 1;
    M_right.at<double>(0, 0) = 1;
    M_right.at<double>(1, 1) = 1;
    M_right.at<double>(2, 2) = 1;
    M_right.at<double>(0, 3) = -baseline;

    M_left = K * M_left;
    M_right = K * M_right;

    Mat landmarks;

    triangulatePoints(M_left, M_right, feature_p1, feature_p2, landmarks);

    std::vector<Point3f> output;

    for (int i = 0; i < landmarks.cols; i++) {
        Point3f p;
        p.x = landmarks.at<float>(0, i) / landmarks.at<float>(3, i);
        p.y = landmarks.at<float>(1, i) / landmarks.at<float>(3, i);
        p.z = landmarks.at<float>(2, i) / landmarks.at<float>(3, i);
        output.push_back(p);
    }

    return output;
}

void VO::extract_keypoints_surf(const Mat &img1, const Mat &img2, vector<Point3f> &landmarks,
                                vector<Point2f> &feature_points) {
    assert(landmarks.empty());
    assert(feature_points.empty());

    // Feature Detection and Extraction
    Ptr<FeatureDetector> detector = ORB::create(350);

    vector<KeyPoint> keypoints1, keypoints2;
    detector->detect(img1, keypoints1);
    detector->detect(img2, keypoints2);

    // cv::SurfDescriptorExtractor extractor;
    // cv::Ptr<cv::SURF> extractor = cv::SURF::create();
    Ptr<DescriptorExtractor> extractor = ORB::create();

    cv::Mat descriptors1, descriptors2;
    extractor->compute(img1, keypoints1, descriptors1);
    extractor->compute(img2, keypoints2, descriptors2);

    descriptors1.convertTo(descriptors1, CV_32F);
    descriptors2.convertTo(descriptors2, CV_32F);
    // cv::BruteForceMatcher<L2<float> > matcher;

    FlannBasedMatcher matcher;
    vector<vector<DMatch>> matches;
    matcher.knnMatch(descriptors1, descriptors2, matches, 2);

    vector<Point2f> match_points1;
    vector<Point2f> match_points2;

    // std::vector<DMatch> bestMatches;

    for (auto &matche : matches) {
        const DMatch &bestMatch = matche[0];
        const DMatch &betterMatch = matche[1];

        if (bestMatch.distance < 0.7 * betterMatch.distance) {
            match_points1.push_back(keypoints1[bestMatch.queryIdx].pt);
            match_points2.push_back(keypoints2[bestMatch.trainIdx].pt);
            // bestMatches.push_back(bestMatch);
        }
    }

    feature_points = match_points1;

    landmarks = get3D_Points(match_points1, match_points2);

    // drawing the results
    // namedWindow("matches", 1);
    // Mat img_matches;
    // drawMatches(img1, keypoints1, img2, keypoints2, bestMatches, img_matches);
    // imshow("matches", img_matches);
    // waitKey(0);
}


void VO::create_new_features(const cv::Mat &leftImg, const cv::Mat &rightImg,
                             std::vector<Point3f> &landmarks, const Mat &inv_transform,
                             std::vector<Point2f> &featurePoints) {
    if (!featurePoints.empty()) {
        featurePoints.clear();
        landmarks.clear();
    }

    vector<Point3f> landmark_3D_new;
    vector<Point2f> reference_2D_new;

    extract_keypoints_surf(leftImg, rightImg, landmark_3D_new, reference_2D_new);

    for (int k = 0; k < landmark_3D_new.size(); k++) {
        const Point3f &pt = landmark_3D_new[k];

        Point3f p;

        p.x = inv_transform.at<double>(0, 0) * pt.x + inv_transform.at<double>(0, 1) * pt.y +
              inv_transform.at<double>(0, 2) * pt.z + inv_transform.at<double>(0, 3);
        p.y = inv_transform.at<double>(1, 0) * pt.x + inv_transform.at<double>(1, 1) * pt.y +
              inv_transform.at<double>(1, 2) * pt.z + inv_transform.at<double>(1, 3);
        p.z = inv_transform.at<double>(2, 0) * pt.x + inv_transform.at<double>(2, 1) * pt.y +
              inv_transform.at<double>(2, 2) * pt.z + inv_transform.at<double>(2, 3);

        if (p.z > 0) {
            landmarks.push_back(p);
            featurePoints.push_back(reference_2D_new[k]);
        }
    }
}

void tracking(const cv::Mat &ref_img, const cv::Mat &curImg, const std::vector<Point2f> &featurePoints,
              const std::vector<Point3f> &landmarks, std::vector<Point3f> &landmarks_ref,
              std::vector<Point2f> &featurePoints_ref) {
    vector<Point2f> nextPts;
    vector<uchar> status;
    vector<float> err;

    calcOpticalFlowPyrLK(ref_img, curImg, featurePoints, nextPts, status, err);

    for (int j = 0; j < status.size(); j++) {
        if (status[j] == 1) {
            featurePoints_ref.push_back(nextPts[j]);
            landmarks_ref.push_back(landmarks[j]);
        }
    }
}

shared_ptr<Point2f> VO::computePose(const cv::Mat &leftImg, const cv::Mat &rightImg,
                                    vector<Point3f> &landmarks, vector<Point2f> &featurePoints) {
    const Mat &curImg = leftImg;

    std::vector<Point3f> landmarks_ref;
    std::vector<Point2f> featurePoints_ref;

    tracking(prevImg, curImg, featurePoints, landmarks, landmarks_ref, featurePoints_ref);

    if (landmarks_ref.empty()) return nullptr;

    Mat dist_coeffs = Mat::zeros(4, 1, CV_64F);
    Mat rvec, tvec;
    vector<int> inliers;

    solvePnPRansac(landmarks_ref, featurePoints_ref, K, dist_coeffs, rvec, tvec, false,
                   100, 8.0, 0.99, inliers);

    if (inliers.size() < 5) return nullptr;

    Mat R_matrix;
    Rodrigues(rvec, R_matrix);
    R_matrix = R_matrix.t();
    Mat t_vec = -R_matrix * tvec;

    Mat inv_transform = Mat::zeros(3, 4, CV_64F);
    R_matrix.col(0).copyTo(inv_transform.col(0));
    R_matrix.col(1).copyTo(inv_transform.col(1));
    R_matrix.col(2).copyTo(inv_transform.col(2));
    t_vec.copyTo(inv_transform.col(3));

    create_new_features(leftImg, rightImg, landmarks, inv_transform, featurePoints);

    curImg.copyTo(prevImg);

    t_vec.convertTo(t_vec, CV_32F);
    cout << t_vec.t() << endl;

    return std::make_shared<Point2f>(t_vec.at<float>(0), t_vec.at<float>(2));
}


//############################################################
// static methods here

std::vector<vector<float>> VO::get_Pose(const std::string &path) {
    std::vector<vector<float>> poses;
    ifstream myfile(path);
    string line;
    if (myfile.is_open()) {
        while (getline(myfile, line)) {
            char *dup = strdup(line.c_str());
            char *token = strtok(dup, " ");
            std::vector<float> v;
            while (token != nullptr) {
                v.push_back(atof(token));
                token = strtok(nullptr, " ");
            }
            poses.push_back(v);
            free(dup);
        }
        myfile.close();
    } else
        cerr << "Unable to open file" << endl;
    return poses;
}



