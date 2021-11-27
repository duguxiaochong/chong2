#ifndef  TRTX_YOLOV5_CAR_MESSAGE_HPP_
#define TRTX_YOLOV5_CAR_MESSAGE_HPP_

#include <iostream>
#include <vector>
#include "common.hpp"
#include "map_info.hpp"

typedef struct Position {			// 车车坐标点
    float x;
    float y;
} Pos;

struct CarPositionSend {			// 套接字内容
    Pos blue1;
    Pos blue2;
    Pos red1;
    Pos red2;

    Pos gray1;
    Pos gray2;
    Pos gray3;
    Pos gray4;
} socketInfo;



typedef struct
{
    cv::Rect    img_r;
    cv::Point2f carPosition;
    int         color;      // 0蓝 1红 3黑
    int         num;        // 1 / 2
} car;


typedef struct
{
    cv::Point   img_center;
    int         color;      // 0蓝 1红 3黑
    int         num;        // 1 / 2
} armor;


std::vector<car>    allCar;
std::vector<armor>  allArmor;


cv::Point2f opt4ToCenter(cv::Point2f pts[4]) {
    cv::Point2f center;
    
    center.x = (pts[0].x + pts[1].x + pts[2].x + pts[3].x) / 4.0;
    center.y = (pts[0].y + pts[1].y + pts[2].y + pts[3].y) / 4.0;

    return center;
}


void classify(std::vector<Yolo::Detection> res, std::vector<bbox_t> detections, cv::Mat img, cv::Mat& warpmatrix) {
    // 分类数据
    // 分到 allCar 和 allArmor
    MapInfo t;
    for (size_t j = 0; j < res.size(); j++) { // res.size() 该图检测到多少个物体
        car      temp_car;
        cv::Rect temp_r;
        temp_r  = get_rect(img, res[j].bbox);

        if (res[j].class_id == 0) {  // 0 'car'
            temp_car.img_r          = temp_r;
            temp_car.carPosition    = t.getTargetPoint(cv::Point2f(temp_r.x+temp_r.width/2, temp_r.y+temp_r.height/2), warpmatrix);
            temp_car.color          = -1;
            temp_car.num            = -1;

            // 矫正
            flip_vertical(temp_car.carPosition.y); // 垂直翻转
            correct_function(temp_car.carPosition.x, temp_car.carPosition.y);
            flip_vertical(temp_car.carPosition.y); // 垂直翻转
            
            // 
            allCar.push_back(temp_car);
        }
    }

    // 上交四点模型
    // 四点     detections[i].pts               [0] [1] [2] [3]
    // 数字     detections[i].tag_id            1   2   3   4   5
    // 颜色     detections[i].color_id          0蓝 1红 2黑
    for (int j = 0; j < detections.size(); j++) {
        armor       temp_armor;
        cv::Point2f img_armor_center;

        // img_center
        img_armor_center = opt4ToCenter(detections[j].pts);
        temp_armor.img_center = img_armor_center;

        // color
        temp_armor.color = detections[j].color_id;  // 0蓝 1红 2黑

        // armor_num
        if( detections[j].tag_id == 2 ) {
            temp_armor.num  = detections[j].tag_id;    // 1 / 2
        }
        else {
            temp_armor.num  = 1;
        }

        //
        allArmor.push_back(temp_armor);
    }
}



// rect.contains(cv::Point(x, y));              //返回布尔变量，判断rect是否包含Point(x, y)点
// center (r.x+r.width/2, r.y+r.height/2)
// 判断 [car 的矩形框[] 中是否有 [armor 的中心点] 存在
void carColorProcess() {
    for (auto i = allCar.begin(); i != allCar.end(); i++) {
        for (auto j = allArmor.begin(); j != allArmor.end(); j++) {
            if( (*i).img_r.contains((*j).img_center) ) {
                (*i).color = (*j).color;
                (*i).num = (*j).num;
                break;
            }
        }
    }
}


void showCarInfo() {
    for (auto i = allCar.begin(); i != allCar.end(); i++) {
        if ((*i).color == 0) {
            std::cout << "    car_color:    blue" << std::endl;
        }
        else if ((*i).color == 1) {
            std::cout << "    car_color:    red" << std::endl;
        }
        else if ((*i).color == 2) {
            std::cout << "    car_color:    gray" << std::endl;
        }
        else if ((*i).color == -1) {
            std::cout << "    car_color:    I don't know" << std::endl;
        }
        std::cout << "      car_num:    " << (*i).num << std::endl;
        std::cout << "carPosition_x:    " << (*i).carPosition.x << std::endl;
        std::cout << "carPosition_y:    " << (*i).carPosition.y << std::endl;
        std::cout << std::endl;
    }
    allCar.clear();
    allArmor.clear();
}


// reutrn 当前图片 内容
struct CarPositionSend tempToSendmessage () {

 // Init == > ocketInfo
socketInfo.blue1.x = -1;
socketInfo.blue1.y = -1;
socketInfo.blue2.x = -1;
socketInfo.blue2.y = -1;
socketInfo.red1.x = -1;
socketInfo.red1.y = -1;
socketInfo.red2.x = -1;
socketInfo.red2.y = -1;
socketInfo.gray1.x = -1;
socketInfo.gray1.y = -1;
socketInfo.gray2.x = -1;
socketInfo.gray2.y = -1;
socketInfo.gray3.x = -1;
socketInfo.gray3.y = -1;
socketInfo.gray4.x = -1;
socketInfo.gray4.y = -1;

    int gray_num = 0;
    for (auto i = allCar.begin(); i != allCar.end(); i++) {
        if ( (*i).color == 0 ) {    // blue
            if( (*i).num == 1 ) {
                socketInfo.blue1.x = (*i).carPosition.x;
                socketInfo.blue1.y = (*i).carPosition.y;
            }
            else if ( (*i).num == 2 ) {
                socketInfo.blue2.x = (*i).carPosition.x;
                socketInfo.blue2.y = (*i).carPosition.y;
            }
        }   
        else if ( (*i).color == 1 ) { // red
            if( (*i).num == 1 ) {
                socketInfo.red1.x = (*i).carPosition.x;
                socketInfo.red1.y = (*i).carPosition.y;
            }
            else if ( (*i).num == 2 ) {
                socketInfo.red2.x = (*i).carPosition.x;
                socketInfo.red2.y = (*i).carPosition.y;
            }
        }
        else if ( (*i).color == 2 ) { // gray
            gray_num ++;
            if( gray_num == 1) {
                socketInfo.gray1.x = (*i).carPosition.x;
                socketInfo.gray1.y = (*i).carPosition.y;
            }
            else if (gray_num == 2) {
                socketInfo.gray2.x = (*i).carPosition.x;
                socketInfo.gray2.y = (*i).carPosition.y;
            }
            else if (gray_num ==3) {
                socketInfo.gray3.x = (*i).carPosition.x;
                socketInfo.gray3.y = (*i).carPosition.y;
            }
            else if (gray_num == 4) {
                socketInfo.gray4.x = (*i).carPosition.x;
                socketInfo.gray4.y = (*i).carPosition.y;
            }
        }
    }
    allCar.clear();
    allArmor.clear();

    return socketInfo;
}



#endif  // TRTX_YOLOV5_CAR_MESSAGE_HPP_