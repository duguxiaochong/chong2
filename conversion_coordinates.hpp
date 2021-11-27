#ifndef TRTX_YOLOV5_CONVERSION_COORDINATES_HPP_
#define TRTX_YOLOV5_CONVERSION_COORDINATES_HPP_

// 把 [图片坐标系] 转化为 [哨岗坐标系] 所需转换function

#define MAP_WIDTH   448
#define MAP_HIGH    808

#include <iostream>

// 水平翻转
void flip_horizontal(float& x) {
    x = MAP_WIDTH - x;
}

// 垂直翻转
void flip_vertical(float& y) {
    y = MAP_HIGH - y;
}

// 对角翻转
void flip_diagonal(float& x, float& y) {
    flip_horizontal(x);
    flip_vertical(y);
}

#endif // TRTX_YOLOV5_CONVERSION_COORDINATES_HPP_