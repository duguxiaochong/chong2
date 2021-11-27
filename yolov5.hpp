#include <iostream>
#include <chrono>
#include <cmath>
#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"
#include "utils.h"
#include "calibrator.h"

#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32           // 构建 .engine 模型的时候用的，可以注释掉了，但我懒
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1


// stuff we know about the network and the input/output blobs
static const int INPUT_H        = Yolo::INPUT_H;
static const int INPUT_W        = Yolo::INPUT_W;
static const int CLASS_NUM      = Yolo::CLASS_NUM;                  // 构建 .engine 模型的时候用的，可以注释掉了，但我懒
static const int OUTPUT_SIZE    = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME     = "data";
const char* OUTPUT_BLOB_NAME    = "prob";
static Logger gLogger;

int                 inputIndex;
int                 outputIndex;
size_t              size = 0;
char*               trtModelStream;
static float        data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
static float        prob[BATCH_SIZE * OUTPUT_SIZE];
IRuntime*           runtime;
ICudaEngine*        engine;
IExecutionContext*  context;
void*               buffers[2];
cudaStream_t        stream;

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
void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
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
    // 为了绑定缓冲区, 我们需要知道输入和输出张量的名称.
    // 注意索引保证小于 IEngine::getbbindings()
    inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    // 在设备上创建GPU缓冲区
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    // Create stream
    CUDA_CHECK(cudaStreamCreate(&stream));
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

    cv::Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H); // letterbox BGR to RGB     // 预处理 BGR2RGB
    int i = 0;
    for (int row = 0; row < INPUT_H; ++row) {
        uchar *uc_pixel = pr_img.data + row * pr_img.step;
        for (int col = 0; col < INPUT_W; ++col) {
            data[0 * 3 * INPUT_H * INPUT_W + i] = (float)uc_pixel[2] / 255.0;
            data[0 * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
            data[0 * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
            uc_pixel += 3;
            ++i;
        }
    }

    // 推理 以及 输出推理时间
    auto start = std::chrono::system_clock::now();
    doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    std::vector<std::vector<Yolo::Detection>> batch_res(1);
    // GG
    auto &res = batch_res[0];
    nms(res, &prob[0 * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);

    return batch_res[0];
}