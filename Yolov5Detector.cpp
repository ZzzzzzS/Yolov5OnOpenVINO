#include "Yolov5Detector.h"

YoloDetector::YoloDetector(std::string ModelPath,std::string device,double conf,double nms_thres)
    :confidence(conf),
    nms_threshold(nms_thres)
{
    this->Network = this->Core.ReadNetwork(ModelPath);
	this->input_info = this->Network.getInputsInfo();
	this->InputName = this->input_info.begin()->first; //获取输入的名称
	this->ImageH = this->Network.getInputShapes().begin()->second[2];
	this->ImageW = this->Network.getInputShapes().begin()->second[3];

	this->output_info = this->Network.getOutputsInfo();
	for (auto &item : output_info)
	{
	}

	this->ExecNetwork = this->Core.LoadNetwork(ModelPath, device); //默认在GPU上跑
	this->InferRequest = this->ExecNetwork.CreateInferRequest();
    this->ScaledImage=cv::Mat::zeros(this->ImageH,this->ImageW,CV_8UC3);
}

YoloDetector::~YoloDetector()
{

}

void YoloDetector::AddAnchors(std::string LayerName, std::vector<cv::Point2d> anchors)
{
    this->anchors.insert(std::pair<std::string, std::vector<cv::Point2d>>(LayerName, anchors));
}

bool YoloDetector::DetectOnce(cv::Mat &Src, std::vector<DetectedObject> &result)
{
    result.clear();
	cv::Rect ScaledImageSize;
    this->PreProcessImage(Src,this->ScaledImage, this->ImageW, this->ImageH, ScaledImageSize);
	cv::Mat blob_image = this->ScaledImage;
	auto BlobIn = this->InferRequest.GetBlob(this->InputName);

	float* data = static_cast<float*>(BlobIn->buffer());

	auto ImageSize = this->ImageH*this->ImageW;
	auto BlobImagePtr = blob_image.ptr<uchar>(0);
	//三通道分离重新排序
	for (size_t chennel = 0; chennel < 3; chennel++) //通道
	{
		for (size_t j = 0; j < ImageSize; j++) //图片像素
		{
			data[chennel*ImageSize + j] = (float)(BlobImagePtr[j*3 + chennel]) / 255.0;
		}
	}

	this->InferRequest.Infer(); //执行推测
	int count = 0;
	std::vector<cv::Rect2d> obj;
	std::vector<float> conf;
	std::vector<int> types;
	for (auto& i : this->output_info)
	{
		auto output = this->InferRequest.GetBlob(i.first);
		this->PostInfer(output, obj, conf, types, i.first);
		count++;
	}

	if (obj.empty()) //检测不出来就返回false
		return false;

	std::vector<int> index;
	cv::dnn::NMSBoxes(obj, conf, this->confidence, this->nms_threshold, index);
	
	if (index.empty())//检测不出来就返回false
		return  false;

	for (auto i : index)
	{ 
		DetectedObject temp;
		//temp.x = (obj[i].x-(double)ScaledImageSize.x)/(double)this->ImageW; //这个地方应该会有问题，但是好像没问题
		//temp.y = (obj[i].y-(double)ScaledImageSize.y)/(double)this->ImageH;
		temp.x = (obj[i].x) / (double)this->ImageW;
		temp.y = (obj[i].y) / (double)this->ImageH;
		temp.w = obj[i].width/this->ImageW;
		temp.h = obj[i].height/this->ImageH;
		temp.conf = conf[i];
		temp.type = types[i];
		result.push_back(temp);
	}
	return true;
}

bool YoloDetector::DetectSingle(cv::Mat& Src, int type, DetectedObject& result)
{
	std::vector<DetectedObject> DetectedObject;
	if(!this->DetectOnce(Src,DetectedObject))
		return false;

	double highestConf=-1;
	for (auto& item : DetectedObject)
	{
		if (item.type != type)
			continue;
		if (item.conf > highestConf)
		{
			highestConf = item.conf;
			result = item;
		}
	}
	if (highestConf == -1)
		return false;

	return true;

}

double YoloDetector::sigmoid(double x)
{
    return (1 / (1 + exp(-x)));
}

void YoloDetector::PreProcessImage(cv::Mat& src,cv::Mat& dst, int input_w, int input_h, cv::Rect& ROI)
{
    int w, h, x, y;
	float r_w = input_w / (src.cols*1.0);
	float r_h = input_h / (src.rows*1.0);
	if (r_h > r_w) {
		w = input_w;
		h = r_w * src.rows;
		x = 0;
		y = (input_h - h) / 2;
	}
	else {
		w = r_h * src.cols;
		h = input_h;
		x = (input_w - w) / 2;
		y = 0;
	}

    this->ScaledImageTemp=cv::Mat(h,w,CV_8UC3);
	cv::resize(src, ScaledImageTemp, ScaledImageTemp.size(), 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(this->ScaledImageTemp,this->ScaledImageTemp,cv::COLOR_BGR2RGB);

	this->ScaledImage=cv::Mat(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));

	this->ScaledImageTemp.copyTo(this->ScaledImage(cv::Rect(x, y, ScaledImageTemp.cols, ScaledImageTemp.rows)));
	ROI.x = x;
	ROI.y = y;
	ROI.width = ScaledImageTemp.cols;
	ROI.height = ScaledImageTemp.rows;
}

void YoloDetector::PostInfer(InferenceEngine::Blob::Ptr blob, std::vector<cv::Rect2d>& obj, std::vector<float>& conf, std::vector<int>& types, std::string LayerName)
{
    float* output_blob = static_cast<float*>(blob->buffer());
	auto dim = blob->getTensorDesc().getDims();

	int item_size = dim[4];
	int anchor_n = dim[1];
	int net_grid = dim[2];

	for (int n = 0; n < anchor_n; ++n)
		for (int i = 0; i < net_grid; ++i)
			for (int j = 0; j < net_grid; ++j)
			{
				double box_prob = output_blob[n*net_grid*net_grid*item_size + i * net_grid*item_size + j * item_size + 4];
				box_prob = sigmoid(box_prob);
				//框置信度不满足则整体置信度不满足
				if (box_prob < this->confidence)
					continue;

				//注意此处输出为中心点坐标,需要转化为角点坐标
				double x = output_blob[n*net_grid*net_grid*item_size + i * net_grid*item_size + j * item_size + 0];
				double y = output_blob[n*net_grid*net_grid*item_size + i * net_grid*item_size + j * item_size + 1];
				double w = output_blob[n*net_grid*net_grid*item_size + i * net_grid*item_size + j * item_size + 2];
				double h = output_blob[n*net_grid*net_grid*item_size + i * net_grid*item_size + j * item_size + 3];

				double max_prob = 0;
				int idx = 0;
				for (int t = 5; t < item_size; ++t) {
					double tp = output_blob[n*net_grid*net_grid*item_size + i * net_grid*item_size + j * item_size + t];
					tp = sigmoid(tp);
					if (tp > max_prob) {
						max_prob = tp;
						idx = t;
					}
				}
				float cof = box_prob * max_prob;
				//对于边框置信度小于阈值的边框,不关心其他数值,不进行计算减少计算量
				if (cof < this->confidence)
					continue;

				auto anchor = this->anchors[LayerName];

				x = (sigmoid(x) * 2 - 0.5 + j) / net_grid * this->ImageH;
				y = (sigmoid(y) * 2 - 0.5 + i) / net_grid * this->ImageW;
				w = pow(sigmoid(w) * 2, 2) * anchor[n].x;
				h = pow(sigmoid(h) * 2, 2) * anchor[n].y;

				double r_x = x - w / 2;
				double r_y = y - h / 2;

				cv::Rect2d temp;
				temp.x = r_x;
				temp.y = r_y;
				temp.width = w;
				temp.height = h;
				obj.push_back(temp);
				conf.push_back(cof);
				types.push_back(idx - 5);
			}
}


void YoloDetector::DrawRectangle(cv::Mat &src, std::vector<DetectedObject> &results, bool TextOn)
{
    for (auto i : results)
	{
		cv::Rect r;
		r.x = i.x*src.cols;
		r.y = i.y*src.rows;
		r.width = i.w*src.cols;
		r.height = i.h*src.rows;

		cv::rectangle(src, r, cv::Scalar(0x27, 0xC1, 0x36), 5);
		cv::putText(src, std::to_string(i.conf*100)+"%", cv::Point(r.x+r.width, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
		if(TextOn)
			cv::putText(src, std::to_string(i.type), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
	}
}