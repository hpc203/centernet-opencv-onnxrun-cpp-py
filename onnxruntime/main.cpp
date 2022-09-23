#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>

using namespace std;
using namespace cv;
using namespace Ort;

struct Object
{
	cv::Rect rect;
	int label;
	float score;
};

struct Net_config
{
	float confThreshold; // Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	string modelpath;
};

class CenterNet
{
public:
	CenterNet(Net_config config);
	Mat detect(Mat& frame);
private:
	int inpWidth;
	int inpHeight;
	vector<string> class_names;
	int num_class;
	int num_grid_x;
	int num_grid_y;

	float confThreshold;
	float nmsThreshold;
	void nms(vector<Object>& input_boxes);
	const float mean[3] = { 0.406, 0.456, 0.485 };
	const float std[3] = { 0.225, 0.224, 0.229 };

	vector<float> input_image_;
	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "YOLOPv2");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
	void copy_data(Mat img);
};

CenterNet::CenterNet(Net_config config)
{
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;

	string model_path = config.modelpath;
	std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
	//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, widestr.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];
	this->num_grid_y = output_node_dims[0][2];
	this->num_grid_x = output_node_dims[0][3];

	string classesFile = "coco.names";
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) this->class_names.push_back(line);
	this->num_class = class_names.size();
}

void CenterNet::copy_data(Mat img)
{
	//    img.convertTo(img, CV_32F);
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(row * col * img.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				this->input_image_[c * row * col + i * col + j] = img.ptr<float>(i)[j * 3 + c];
			}
		}
	}
}

inline float sigmoid(float x)
{
	return 1.0 / (1.0 + expf(-x));
}

void CenterNet::nms(vector<Object>& inputBoxes)
{
	std::sort(inputBoxes.begin(), inputBoxes.end(), [](Object a, Object b) { return a.score > b.score; });

	std::vector<bool> isSuppressed(inputBoxes.size(), false);
	for (int i = 0; i < int(inputBoxes.size()); ++i)
	{
		if (isSuppressed[i])
		{
			continue;
		}
		for (int j = i + 1; j < int(inputBoxes.size()); ++j)
		{
			if (isSuppressed[j])
			{
				continue;
			}

			float inter = (inputBoxes[i].rect&inputBoxes[j].rect).area();
			float ovr = inter / (inputBoxes[i].rect.area() + inputBoxes[j].rect.area() - inter);

			if (ovr >= this->nmsThreshold)
			{
				isSuppressed[j] = true;
			}
		}
	}

	// return post_nms;
	int idx_t = 0;
	inputBoxes.erase(remove_if(inputBoxes.begin(), inputBoxes.end(), [&idx_t, &isSuppressed](const Object &f) { return isSuppressed[idx_t++]; }), inputBoxes.end());
}

Mat CenterNet::detect(Mat& frame)
{
	Mat dstimg;
	resize(frame, dstimg, Size(this->inpWidth, this->inpHeight));
	cvtColor(dstimg, dstimg, cv::COLOR_BGR2RGB);
	vector<Mat> rgbChannels(3);
	split(dstimg, rgbChannels);
	for (int i = 0; i < rgbChannels.size(); i++)
	{
		rgbChannels[i].convertTo(rgbChannels[i], CV_32FC1, 1.0 / (255.0*std[i]), (0.0 - mean[i]) / std[i]);
	}
	merge(rgbChannels, dstimg);
	this->copy_data(dstimg);
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	// 开始推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // 开始推理

	/////generate proposals
	vector<Object> generate_boxes;
	float ratioh = (float)frame.rows / this->inpHeight, ratiow = (float)frame.cols / this->inpWidth;
	const float stride = this->inpHeight / num_grid_y;


	const float* pscore = ort_outputs[0].GetTensorMutableData<float>();
	const float* pxy = ort_outputs[1].GetTensorMutableData<float>();
	const float* pwh = ort_outputs[2].GetTensorMutableData<float>();
	int i = 0, j = 0, c = 0, area = num_grid_y * num_grid_x;
	for (i = 0; i < num_grid_y; i++)
	{
		for (j = 0; j < num_grid_x; j++)
		{
			float max_class_score = -1000;
			int class_id = -1;
			for (c = 0; c < num_class; c++)
			{
				float score = sigmoid(pscore[c * area + i * num_grid_x + j]);
				if (score > max_class_score)
				{
					max_class_score = score;
					class_id = c;
				}
			}

			if (max_class_score > confThreshold)
			{
				float cx = (pxy[i * num_grid_x + j] + j) * stride * ratiow;  ///cx
				float cy = (pxy[area + i * num_grid_x + j] + i) * stride * ratioh;   ///cy
				float w = pwh[i * num_grid_x + j] * stride * ratiow;   ///w
				float h = pwh[area + i * num_grid_x + j] * stride * ratioh;  ///h

				int x = max(int(cx - 0.5*w), 0);
				int y = max(int(cy - 0.5*h), 0);
				int width = min(int(w), frame.cols - 1);
				int height = min(int(h), frame.rows - 1);
				generate_boxes.push_back(Object{ Rect(x, y, width, height),  class_id, max_class_score });
			}
		}
	}
	nms(generate_boxes);

	Mat outimg = frame.clone();
	for (i = 0; i < generate_boxes.size(); ++i)
	{
		int xmin = generate_boxes[i].rect.x;
		int ymin = generate_boxes[i].rect.y;
		rectangle(outimg, Point(xmin, ymin), Point(xmin + generate_boxes[i].rect.width, ymin + generate_boxes[i].rect.height), Scalar(0, 0, 255), 2);
		string label = format("%.2f", generate_boxes[i].score);
		label = this->class_names[generate_boxes[i].label] + ":" + label;
		putText(outimg, label, Point(xmin, ymin - 5), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
	}
	return outimg;
}

int main()
{
	Net_config cfg = { 0.4, 0.5, "ctdet_coco_dlav0_384.onnx" };   ////choices = ["ctdet_coco_dlav0_384.onnx", "ctdet_coco_dlav0_512.onnx"]
	CenterNet net(cfg);
	string imgpath = "images/person.jpg";
	Mat srcimg = imread(imgpath);
	Mat outimg = net.detect(srcimg);

	//imwrite("result.jpg", outimg);

	static const string kWinName = "Deep learning object detection in ONNXRuntime";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, outimg);
	waitKey(0);
	destroyAllWindows();
}
