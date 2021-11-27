cmake_minimum_required(VERSION 3.15)

project(opt4)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})  # CMAKE_CURRENT_SOURCE_DIR --> 这是当前处理的CMakeLists.txt所在的目录

find_package(TensorRT REQUIRED)
find_package(OpenCV  REQUIRED)
# find_package(Eigen3 REQUIRED)     # 上交预测用的
# find_package(Ceres 2 REQUIRED)    # 上交预测用的
# find_package(fmt REQUIRED)        # 快速 输出流 库 ?
# include_directories(/usr/local/include/fmt/)
# link_directories(/usr/local/lib/)

include_directories(${OpenCV_INCLUDE_DIRS})
# include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${TensorRT_INCLUDE_DIRS})
# include_directories(${CERES_INCLUDE_DIRS})

add_library(opt4 OBJECT TRTModule.cpp ${TensorRT_SOURCE})
target_link_libraries(opt4 ${OpenCV_LIBS} ${TensorRT_LIBS} nvonnxparser cuda)   # fmt::fmt
