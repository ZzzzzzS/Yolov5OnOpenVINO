#include "Yolov5Detector.h"
#include <opencv2/videoio.hpp>


int main()
{
    YoloDetector Detector("C:/Users/ZhouZishun/OneDrive/documents/TechStuff/PCSoftware/AI/YoloLearning/yolov5/deployment/Yolov5OnOpenVinoOnWindows/yolov5s.onnx");
    std::vector<cv::Point2d> layer1 = { cv::Point(10, 13),cv::Point(16,30),cv::Point(33,23) };
    std::vector<cv::Point2d> layer2 = { cv::Point(30, 61),cv::Point(62,45),cv::Point(59,119) };
    std::vector<cv::Point2d> layer3 = { cv::Point(116, 90),cv::Point(156,198),cv::Point(373,326) };
    Detector.AddAnchors("output", layer3);
    Detector.AddAnchors("667", layer2);
    Detector.AddAnchors("687", layer1);

    cv::VideoCapture* cap=new cv::VideoCapture();
    cap->open("C:/Users/ZhouZishun/Videos/288840339/1/1.mp4");

    std::vector<YoloDetector::DetectedObject> r;

    cv::Mat frame,out;
    while (true)
    {
        if(!cap->read(frame))
            continue;
        Detector.DetectOnce(frame,r);
        Detector.DrawRectangle(frame,r);
        cv::resize(frame,out,cv::Size(0,0),0.5,0.5);
        cv::imshow("result",out);
        cv::waitKey(1);
    }
    


    return 0;
}