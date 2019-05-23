#ifndef VO_H
#define VO_H

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"

class VO {
public:
    cv::Mat prevImg;

    VO() = default;

    virtual ~VO() = default;

    VO(const cv::Mat &_k, float _baseline);

    VO(const cv::Mat &_k, float _baseline, const cv::Mat &firstImg);

    //input is the feature points 1 and feature points 2 in each image, output is the 3D points, stereo images
    std::vector<cv::Point3f> get3D_Points(const std::vector<cv::Point2f> &feature_p1,
                                          const std::vector<cv::Point2f> &feature_p2) const;

    //input  is two stere image pair, output is the feature points of image1 and generated 3D points of landmarks
    void extract_keypoints_surf(const cv::Mat &img1, const cv::Mat &img2,
                                std::vector<cv::Point3f> &landmarks,
                                std::vector<cv::Point2f> &feature_points);

    //input start, inv_transform is the transformation matrix,  output is featurePoints-2D points of features, landmarks is 3D points of landmarkds
    void create_new_features(const cv::Mat &leftImg, const cv::Mat &rightImg, std::vector<cv::Point3f> &landmarks,
                             const cv::Mat &inv_transform, std::vector<cv::Point2f> &featurePoints);

    //this is to play on the image sequence
    std::shared_ptr<cv::Point2f>
    computePose(const cv::Mat &leftImg, const cv::Mat &rightImg, std::vector<cv::Point3f> &landmarks,
                std::vector<cv::Point2f> &featurePoints);

    cv::Mat getK() const { return K; }

    void setK(const cv::Mat &_K) { K = _K; }

    float getBaseline() const { return baseline; }

    void setBaseline(float _baseline) { baseline = _baseline; }

    //get the true pose files and convert it into vector form
    static std::vector<std::vector<float>> get_Pose(const std::string &path);

private:
    cv::Mat K;
    float baseline;
};

#endif