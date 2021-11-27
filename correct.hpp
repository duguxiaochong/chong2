#ifndef TRTX_YOLOV5_CORRECT_HPP_
#define TRTX_YOLOV5_CORRECT_HPP_

#include <iostream>

float watchDog  = 1710.0;
//1768
float carHeight = 250.0;
float nicetry = carHeight / watchDog;

void correct_function(float& x, float& y) {
    x = x * (1.0 - nicetry);
    y = y * (1.0 - nicetry);
}

#endif  // TRTX_YOLOV5_CORRECT_HPP_