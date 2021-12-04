cmake_minimum_required(VERSION 3.5)
project(Yolov5OnOpenVINO)
set(CMAKE_CXX_STANDARD 11)

include_directories(
    #OpenVINO推理引擎的头文件
    "C:/Program Files (x86)/Intel/openvino_2021/deployment_tools/inference_engine/include"
    "C:/Program Files (x86)/Intel/openvino_2021/deployment_tools/ngraph/include/"
    "C:/Program Files (x86)/Intel/openvino_2021.4.752/deployment_tools/ngraph/include"
    "C:/Program Files (x86)/Intel/openvino_2021/opencv/include"
)

#查找必要的依赖包
set(OpenCV_DIR "C:/Program Files (x86)/Intel/openvino_2021/opencv/cmake")
find_package(OpenCV REQUIRED)
set(InferenceEngine_DIR "C:/Program Files (x86)/Intel/openvino_2021.4.752/deployment_tools/inference_engine/share")
find_package(InferenceEngine)
set(ngraph_DIR "C:/Program Files (x86)/Intel/openvino_2021.4.752/deployment_tools/ngraph/cmake")
find_package(ngraph REQUIRED)

set(SOURCES 
        main.cpp 
        Yolov5Detector.h
        Yolov5Detector.cpp)

add_executable(Yolov5OnOpenVINO ${SOURCES})
target_link_libraries(Yolov5OnOpenVINO
        ${OpenCV_LIBS}
        ${InferenceEngine_LIBRARIES}
        ${ngraph_LIBRARIES}
)