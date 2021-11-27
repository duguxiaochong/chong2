#include <iostream>
#include <chrono>
#include <cmath>
#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"
#include "utils.h"
#include "calibrator.h"
#include "preprocess.h"

#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32           // 构建 .engine 模型的时候用的，可以注释掉了，但我懒
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1
#define MAX_IMAGE_INPUT_SIZE_THRESH 3000 * 3000 // ensure it exceed the maximum size in the input images !


// stuff we know about the network and the input/output blobs
static const int INPUT_H        = Yolo::INPUT_H;
static const int INPUT_W        = Yolo::INPUT_W;
static const int CLASS_NUM      = Yolo::CLASS_NUM;                  // 构建 .engine 模型的时候用的，可以注释掉了，但我懒
static const int OUTPUT_SIZE    = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME     = "data";
const char* OUTPUT_BLOB_NAME    = "prob";
static Logger gLogger;


char*                   trtModelStream;
size_t                  size;
static float            prob[BATCH_SIZE * OUTPUT_SIZE];
IRuntime*               runtime;
ICudaEngine*            engine;
IExecutionContext*      context;
float*                  buffers[2];                         // change<----
int                     inputIndex;
int                     outputIndex;
cudaStream_t            stream;
uint8_t*                img_host;                           // new<----
uint8_t*                img_device;                         // new<----
// static float            data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
std::vector<cv::Mat>    imgs_buffer(BATCH_SIZE);           // new<----



// start推理函数 用的 error信息, 为什么报错说 `此声明没有存储类或类型说明符` 呢
std::vector<Yolo::Detection>    error(1);
Yolo::Detection                 error_info;
// error_info.bbox[0]  = -1.0;
// error_info.bbox[1]  = -1.0;
// error_info.bbox[2]  = -1.0;
// error_info.bbox[3]  = -1.0;
// error_info.conf     = -1.0;
// error_info.class_id = -1.0;
// error.push_back(error_info);

// 推理
void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* output, int batchSize) {
    // infer on the batch asynchronously, and DMA output back to host
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

void init(){
    // prepare input data
    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void**)&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    CUDA_CHECK(cudaStreamCreate(&stream));
    img_host = nullptr;
    img_device = nullptr;
    // prepare input data cache in pinned memory 
    CUDA_CHECK(cudaMallocHost((void**)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    // prepare input data cache in device memory
    CUDA_CHECK(cudaMalloc((void**)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
}

void destroy() {
    // Release stream and buffers
    // 释放 流 和 缓冲区
    // cap.release();
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
    // Destroy the engine
    // 销毁 engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
}

// 构造函数
int start(std::string& engine_name) {
    cudaSetDevice(DEVICE);

    // deserialize the .engine and run inference
    // 对 `engine_name` 的处理
    // 对.engine进行反序列化并运行推理
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        return -1;
    }
    
    // trtModelStream, size
    trtModelStream = nullptr;
    size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    init();

    return 0;
}

std::vector<Yolo::Detection> yolov5_v5_Rtx_start(cv::Mat img) {
    if (img.empty()) {
        std::cout << "Fail to read image from camera!" << std::endl;
        // return error;
    }

    float* buffer_idx = (float*)buffers[inputIndex];    // new<----
    imgs_buffer[0] = img;                               // new<----
    size_t size_image = img.cols * img.rows * 3;
    size_t size_image_dst = INPUT_H * INPUT_W * 3;
    // copy data to pinned memory
    memcpy(img_host, img.data, size_image);             // new<----
    // copy data to device memory
    CUDA_CHECK(cudaMemcpyAsync(img_device, img_host, size_image, cudaMemcpyHostToDevice, stream));  // new<----
    preprocess_kernel_img(img_device, img.cols, img.rows, buffer_idx, INPUT_W, INPUT_H, stream);    // new<----
    buffer_idx += size_image_dst;                                                                   // new<----

    // 推理 以及 输出推理时间
    auto start = std::chrono::system_clock::now();
    doInference(*context, stream, (void**)buffers, prob, BATCH_SIZE);
    auto end = std::chrono::system_clock::now();
    std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    std::vector<std::vector<Yolo::Detection>> batch_res(1);
    // GG
    auto &res = batch_res[0];
    nms(res, &prob[0 * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);

    return batch_res[0];
}