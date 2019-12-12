#include <stdio.h>
#include <vector>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#define MODEL_NAME RESOURCE_DIR"/mobilenetv2-1.0.onnx"	// NCHW
#define LABEL_NAME RESOURCE_DIR"/synset.txt"
#define MODEL_WIDTH 224
#define MODEL_HEIGHT 224
#define MODEL_CHANNEL ncnn::Mat::PIXEL_BGR
static const float PIXEL_MEAN[3] = { 0.485f, 0.456f, 0.406f };
static const float PIXEL_STD[3] = { 0.229f, 0.224f, 0.225f };
#define LOOP_NUM_TO_MEASURE_INFERENCE_TIME 100

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

	/* Read image */
	cv::Mat image = cv::imread(RESOURCE_DIR"/parrot.jpg");
	cv::imshow("Display", image);

	/* Repeat process for timea measurement */
	auto t0 = std::chrono::system_clock::now();
	for (int i = 0; i < LOOP_NUM_TO_MEASURE_INFERENCE_TIME; i++) {
		/* Pre-Process */
		cv::resize(image, image, cv::Size(224, 224));
		cv::Mat imageFloat = cv::Mat(image.rows, image.cols, CV_32FC3);
#pragma omp parallel for
		for (int i = 0; i < image.cols * image.rows; i++) {
			for (int c = 0; c < image.channels(); c++) {
				((float*)(imageFloat.data))[image.channels() * i + c] = (float)(((image.data[image.channels() * i + c] / 255.0) - PIXEL_MEAN[c]) / PIXEL_STD[c]);
			}
		}

		/* Inference */
		// 4-dimensional Mat with NCHW
		cv::Mat input = cv::dnn::blobFromImage(imageFloat);
		net.setInput(input);
		std::vector<cv::Mat> outs;
		net.forward(outs, outNames);

		/* Post-Process */
		std::vector<float> scores(outs[0].rows * outs[0].cols);
		scores.assign((float*)outs[0].data, (float*)outs[0].data + scores.size());
		int maxIndex = std::max_element(scores.begin(), scores.end()) - scores.begin();
		float maxScore = *std::max_element(scores.begin(), scores.end());	// todo softmax
		if (i == 0) printf("Result = %s (%.3f)\n", labels[maxIndex].c_str(), maxScore);
	}
	auto t1 = std::chrono::system_clock::now();

	double inferenceTime = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
	printf("Inference time: %.2lf [msec]\n", inferenceTime / LOOP_NUM_TO_MEASURE_INFERENCE_TIME);

	cv::waitKey(0);
	return 0;
}
