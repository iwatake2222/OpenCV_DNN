/*** Include ***/
/* for general */
#include <stdint.h>
#include <stdio.h>
#include <fstream> 
#include <vector>
#include <string>
#include <chrono>

/* for OpenCV */
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

/*** Macro ***/
/* Model parameters */
#define MODEL_NAME RESOURCE"/mobilenetv2-1.0.onnx"	// NCHW
#define LABEL_NAME RESOURCE"/synset.txt"
#define MODEL_WIDTH 224
#define MODEL_HEIGHT 224
static const float PIXEL_MEAN[3] = { 0.485f, 0.456f, 0.406f };
static const float PIXEL_STD[3] = { 0.229f,  0.224f, 0.225f };

/* Settings */
#define LOOP_NUM_FOR_TIME_MEASUREMENT 100


/*** Function ***/
static void readLabel(const char* filename, std::vector<std::string> &labels)
{
	std::ifstream ifs(filename);
	if (ifs.fail()) {
		printf("failed to read %s\n", filename);
		return;
	}
	std::string str;
	while (getline(ifs, str)) {
		labels.push_back(str);
	}
}

int main()
{
	/*** Initialize ***/
	/* read label */
	std::vector<std::string> labels;
	readLabel(LABEL_NAME, labels);

	/* Create network */
	cv::dnn::Net net = cv::dnn::readNetFromONNX(MODEL_NAME);
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	// net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	// net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
	// net.setPreferableTarget(cv::dnn::DNN_TARGET_VULKAN);
	std::vector<cv::String> outNames = net.getUnconnectedOutLayersNames();
	cv::setNumThreads(4);

	/*** Process for each frame ***/
	/* Read image */
	cv::Mat originalImage = cv::imread(RESOURCE"/parrot.jpg");
	cv::Mat inputImage;

	/** Pre-process and Set data to input tensor **/
	cv::resize(originalImage, inputImage, cv::Size(MODEL_WIDTH, MODEL_HEIGHT));
	cv::cvtColor(inputImage, inputImage, cv::COLOR_BGR2RGB);
	/* most of data values becomes -1.0 ~ 1.0 */
	inputImage.convertTo(inputImage, CV_32FC3, 1.0 / 255);
	cv::subtract(inputImage, cv::Scalar(cv::Vec<float, 3>(PIXEL_MEAN)), inputImage);
	cv::divide(inputImage, cv::Scalar(cv::Vec<float, 3>(PIXEL_STD)), inputImage);
//#pragma omp parallel for
//	for (int i = 0; i < inputImage.cols * inputImage.rows; i++) {
//		for (int c = 0; c < originalImage.channels(); c++) {
//			((float*)(inputImage.data))[inputImage.channels() * i + c] = (float)(((inputImage.data[inputImage.channels() * i + c] / 255.0) - PIXEL_MEAN[c]) / PIXEL_STD[c]);
//		}
//	}

	/* 4-dimensional Mat in NCHW */
	cv::Mat input = cv::dnn::blobFromImage(inputImage);
	net.setInput(input);

	/* Run inference */
	std::vector<cv::Mat> outs;
	net.forward(outs, outNames);

	/* Retrieve the result */
	std::vector<float> outputScoreList(outs[0].rows * outs[0].cols);
	outputScoreList.assign((float*)outs[0].data, (float*)outs[0].data + outputScoreList.size());
	int maxIndex = (int)(std::max_element(outputScoreList.begin(), outputScoreList.end()) - outputScoreList.begin());
	auto maxScore = *std::max_element(outputScoreList.begin(), outputScoreList.end());
	printf("%s (%.3f)\n", labels[maxIndex].c_str(), maxScore);
	cv::imshow("test", originalImage); cv::waitKey(1);

	/*** (Optional) Measure inference time ***/
	const auto& t0 = std::chrono::steady_clock::now();
	for (int i = 0; i < LOOP_NUM_FOR_TIME_MEASUREMENT; i++) {
		net.forward(outs, outNames);
	}
	const auto& t1 = std::chrono::steady_clock::now();
	std::chrono::duration<double> timeSpan = t1 - t0;
	printf("Inference time = %f [msec]\n", timeSpan.count() * 1000.0 / LOOP_NUM_FOR_TIME_MEASUREMENT);

	cv::waitKey(-1);

	return 0;
}
