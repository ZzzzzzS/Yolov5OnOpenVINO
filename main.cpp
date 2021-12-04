#include "Yolov5Detector.h"

int main()
{
    YoloDetector Detector("C:/Users/ZhouZishun/Downloads/Projects/yolov5/yolov5-3.1/runs/exp13/weights/best2.onnx");
    std::vector<cv::Point2d> layer1 = { cv::Point(10, 13),cv::Point(16,30),cv::Point(33,23) };
    std::vector<cv::Point2d> layer2 = { cv::Point(30, 61),cv::Point(62,45),cv::Point(59,119) };
    std::vector<cv::Point2d> layer3 = { cv::Point(116, 90),cv::Point(156,198),cv::Point(373,326) };
    Detector.AddAnchors("output", layer3);
    Detector.AddAnchors("655", layer2);
    Detector.AddAnchors("669", layer1);

    cv::Mat src = cv::imread("C:/Users/ZhouZishun/Downloads/Projects/yoloData/FlyingDisk/images/IMG_1687.JPG");
    std::vector<YoloDetector::DetectedObject> r;
    YoloDetector::DetectedObject r2;
    Detector.DetectOnce(src, r);
    Detector.DetectSingle(src, 1, r2);
    Detector.DrawRectangle(src, r, true);
    cv::resize(src,src,cv::Size(640,640));
    cv::imshow("result",src);
    cv::waitKey(0);
    return 0;
}