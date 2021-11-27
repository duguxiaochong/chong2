#ifndef TRTX_YOLOV5_MAP_INGO_HPP_
#define TRTX_YOLOV5_MAP_INGO_HPP_

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>

#include "conversion_coordinates.hpp"
#include "correct.hpp"

class MapInfo
{
private:
    cv::Mat aiMap               = ~ cv::Mat::zeros(808, 448, CV_8UC3);  // 白图
    cv::Mat B3_B7               = ~ cv::Mat::zeros(20,  100, CV_8UC3);
    cv::Mat B1_B4_B6_B9         = ~ cv::Mat::zeros(100, 20,  CV_8UC3);
    cv::Mat B2_B8               = ~ cv::Mat::zeros(80,  20,  CV_8UC3);
    cv::Mat B5                  = cv::Mat::zeros(26,  26,  CV_8UC3);    // 黑图

    cv::Mat aiMapShow           = ~ cv::Mat::zeros(808, 448, CV_8UC3);  // 白图

public:
   MapInfo();
   ~MapInfo();
   void showMapInfo();
   cv::Point2f getTargetPoint(const cv::Point& ptOrigin, const cv::Mat& warpMatrix);
   void drawCarPosition(auto& capCarPosition, const cv::Mat& warpmatrix, cv::Mat img);
};


MapInfo::MapInfo(/* args */) {
    // aiMap = ~aiMap;
    cv::Mat roiB3 = aiMap(cv::Rect(0,       150,      100, 20));    // cv::Rect(左上角的点 和 宽高)
    cv::Mat roiB7 = aiMap(cv::Rect(448-100, 808-170,  100, 20));
    cv::absdiff(roiB3, B3_B7, roiB3);                               // 相减，取绝对值
    cv::absdiff(roiB7, B3_B7, roiB7);

    cv::Mat roiB1 = aiMap(cv::Rect(448-120, 0,        20,  100));
    cv::Mat roiB4 = aiMap(cv::Rect(100,     808-100,  20,  100));
    cv::Mat roiB6 = aiMap(cv::Rect(93,      354,      20,  100));   // 93  -> 93.5
    cv::Mat roiB9 = aiMap(cv::Rect(335,     354,      20,  100));   // 335 -> 334.5
    cv::absdiff(roiB1, B1_B4_B6_B9, roiB1);
    cv::absdiff(roiB4, B1_B4_B6_B9, roiB4);
    cv::absdiff(roiB6, B1_B4_B6_B9, roiB6);
    cv::absdiff(roiB9, B1_B4_B6_B9, roiB9);

    cv::Mat roiB2 = aiMap(cv::Rect(214,     150,      20,  80));
    cv::Mat roiB8 = aiMap(cv::Rect(214,     578,      20,  80));
    cv::absdiff(roiB2, B2_B8, roiB2);
    cv::absdiff(roiB8, B2_B8, roiB8);

    cv::Mat roiB5 = aiMap(cv::Rect(224-13,  404-13,   26,  26));
    // 在B5上画个旋转矩形
    cv::RotatedRect rRect = cv::RotatedRect(cv::Point2f(13,13), cv::Size2f(18, 18), 45);
    cv::Point2f vertices2f[4];      //定义4个点的数组
    rRect.points(vertices2f);       //将四个点存储到vertices数组中
    cv::Point   vertices[4];
    for (int i = 0; i < 4; ++i) {
        vertices[i] = vertices2f[i];
    }
    cv::fillConvexPoly(B5, vertices, 4, cv::Scalar(255,255,255));
    cv::absdiff(roiB5, B5, roiB5);

    aiMap.copyTo(aiMapShow);
}

MapInfo::~MapInfo() {
    
}


void MapInfo::showMapInfo() {
    cv::imshow("map", aiMapShow);
    aiMap.copyTo(aiMapShow);
}


cv::Point2f MapInfo::getTargetPoint(const cv::Point& ptOrigin, const cv::Mat& warpMatrix) {
	cv::Mat_<double> matPt(3, 1);
	matPt(0, 0) = ptOrigin.x;
	matPt(1, 0) = ptOrigin.y;
	matPt(2, 0) = 1;
	cv::Mat matPtView = warpMatrix * matPt;
	double x = matPtView.at<double>(0, 0);
	double y = matPtView.at<double>(1, 0);
	double z = matPtView.at<double>(2, 0);
    
	return cv::Point2f(x * 1.0 / z, y * 1.0 / z);
}

void MapInfo::drawCarPosition(auto& capCarPosition, const cv::Mat& warpmatrix, cv::Mat img) {
    if (capCarPosition.class_id == 0) {     // 当标签为 0, 即 `car` 的时候, 画一个圆代表车车
        cv::Rect r = get_rect(img, capCarPosition.bbox);

        cv::Point2f ptOrigin = cv::Point2f(r.x+r.width/2, r.y+r.height/2);
        cv::Point2f carPosition = getTargetPoint(ptOrigin, warpmatrix);

        cv::circle(aiMapShow, cv::Point(carPosition.x, carPosition.y), 25, cv::Scalar(0, 255, 0));

        flip_vertical(carPosition.y);   // 垂直翻转
        correct_function(carPosition.x, carPosition.y);
        flip_vertical(carPosition.y);   // 垂直翻转
        // std::cout << "carPosition.x" << carPosition.x << std::endl;
        // std::cout << "carPosition.y" << carPosition.y << std::endl;
        cv::circle(aiMapShow, cv::Point(carPosition.x, carPosition.y), 25, cv::Scalar(255, 0, 0));
    }
}

#endif  // TRTX_YOLOV5_TEST_H_255