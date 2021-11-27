cmake_minimum_required(VERSION 2.6)

project(main)

add_definitions(-std=c++17)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

# cuda
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)


include_directories(.)
# include_directories(./TRTX)
include_directories(./opt4)
# add_subdirectory(TRTX)
add_subdirectory(opt4)


add_definitions(-std=c++14)
add_definitions(-DAPI_EXPORTS)
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

cuda_add_library(yolov5 ${PROJECT_SOURCE_DIR}/calibrator.cpp)
target_link_libraries(yolov5 nvinfer)
target_link_libraries(yolov5 cudart)
target_link_libraries(yolov5 myplugins)
target_link_libraries(yolov5 ${OpenCV_LIBS})


add_executable(main yolov5main)
target_link_libraries(main yolov5)    # fmt::fmt
target_link_libraries(main opt4)
target_link_libraries(main -lzmq)
