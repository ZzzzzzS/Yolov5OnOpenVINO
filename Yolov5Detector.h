#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
#include <vector>

class YoloDetector
{
public:
	YoloDetector(std::string ModelPath,std::string device="GPU",double conf=0.5,double nms_thres=0.1);
	~YoloDetector();

public:
	struct DetectedObject
	{
		double conf;
		double x;
		double y;
		double w;
		double h;
		int type;
	};

	//Detect one frame
	bool DetectOnce(cv::Mat &Src, std::vector<DetectedObject> &result);
	//Detect specific class and return the object with highest confidence
	bool DetectSingle(cv::Mat& Src, int type, DetectedObject& result);
	//add anchors
	void AddAnchors(std::string LayerName, std::vector<cv::Point2d> anchors);

	double confidence;
	double nms_threshold;

	static void DrawRectangle(cv::Mat &src, std::vector<DetectedObject> &results, bool TextOn = false);

private:
	InferenceEngine::Core Core;
	InferenceEngine::CNNNetwork Network;
	InferenceEngine::ExecutableNetwork ExecNetwork;

	InferenceEngine::InferRequest InferRequest;

	InferenceEngine::InputsDataMap input_info;
	InferenceEngine::OutputsDataMap output_info;

	std::string InputName;

	int ImageH, ImageW;
	std::map<std::string, std::vector<cv::Point2d>> anchors;

	void PreProcessImage(cv::Mat& src,cv::Mat& dst, int input_w, int input_h, cv::Rect& ROI);
	void PostInfer(InferenceEngine::Blob::Ptr blob, std::vector<cv::Rect2d> &obj, std::vector<float> &conf, std::vector<int> &types, std::string LayerName);
	double sigmoid(double x);

	cv::Mat ScaledImage;
	cv::Mat ScaledImageTemp;
};
