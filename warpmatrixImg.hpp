#ifndef TRTX_YOLOV5_WARPMATRIX_IMG_HPP_
#define TRTX_YOLOV5_WARPMATRIX_IMG_HPP_

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>

// 透视变换矩阵
cv::Mat warpmatrix(3, 3, CV_64FC1);

// 点击事件
static void onMouse1(int event, int x, int y, int, void* userInput) {
    static int          times   = 0;
    static cv::Point2f  fourPoint[4];
    // cv::Mat             src, dst;
	// dst.copyTo(src);
	if (event != cv::EVENT_LBUTTONDOWN) {
        return;    
    }
    else {
        times++;
	    if (times <= 4) {
            std::cout << x << "  " << y << std::endl;
            fourPoint[times-1].x = x;
            fourPoint[times-1].y = y;
        }
        else if (times == 5) {
            //CvMat* mask = cvCreateMat(3, 3, CV_32FC1);
		    // cv::Point2f god_view[] 	  = { cv::Point2f(808,448), cv::Point2f(808,0), cv::Point2f(0,448), cv::Point2f(0,0) };
		    cv::Point2f god_view[] 	    = { cv::Point2f(100, 808-638), cv::Point2f(348, 100), cv::Point2f(120, 708), cv::Point2f(348, 808-150) };
		    //计算变换矩阵
		    warpmatrix = cv::getPerspectiveTransform(fourPoint, god_view);
		    std::cout << warpmatrix << std::endl;
        }
    }
}

// 获取 透视变换 矩阵
cv::Mat getTransformMask(cv::Mat& img) {
    while(true){
        cv::imshow("test", img);
        cv::setMouseCallback("test", onMouse1);
        if (cv::waitKey(1) == 'q') {
            cv::destroyWindow("test");
            break;
        }
    }
    return warpmatrix;
}


// 显示 透视变换 后的图像
void showTransformImg(cv::Mat warpmatrix, cv::Mat img) {
    cv::Mat result;
    cv::warpPerspective(img, result, warpmatrix, cv::Size(448, 808),cv::INTER_LINEAR); //result.size(),
    cv::imshow("result", result);
}



#endif  // TRTX_YOLOV5_TEST_H_