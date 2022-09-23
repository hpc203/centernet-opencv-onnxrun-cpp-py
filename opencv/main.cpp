#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

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

	float confThreshold;
	float nmsThreshold;
	void nms(vector<Object>& input_boxes);

	Net net;
	const float mean[3] = { 0.406, 0.456, 0.485 };
	const float std[3] = { 0.225, 0.224, 0.229 };
};

CenterNet::CenterNet(Net_config config)
{
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;

	this->net = readNet(config.modelpath);

	size_t pos = config.modelpath.rfind("_");
	size_t pos_ = config.modelpath.rfind(".onnx");
	int len = pos_ - pos - 1;
	string hw = config.modelpath.substr(pos + 1, len);
	this->inpHeight = stoi(hw);
	this->inpWidth = stoi(hw);

	string classesFile = "coco.names";
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) this->class_names.push_back(line);
	this->num_class = class_names.size();
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
	Mat blob = blobFromImage(dstimg);
	this->net.setInput(blob);
	vector<Mat> outs;
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());   // 开始推理

	/////generate proposals
	vector<Object> generate_boxes;
	float ratioh = (float)frame.rows / this->inpHeight, ratiow = (float)frame.cols / this->inpWidth;
	const int num_grid_x = outs[0].size[3];
	const int num_grid_y = outs[0].size[2];
	const float stride = this->inpHeight / num_grid_y;


	const float* pscore = (float*)outs[0].data;
	const float* pxy = (float*)outs[1].data;
	const float* pwh = (float*)outs[2].data;
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

	static const string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, outimg);
	waitKey(0);
	destroyAllWindows();
}
