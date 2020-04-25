# Sample project for OpenCV DNN (cv:dnn)

## How to build application code
```
git clone https://github.com/iwatake2222/play_with_opencv_dnn.git
cd play_with_opencv_dnn/
cd onnx_mobilenet_v2/
mkdir build && cd build
cmake ..
make -j4
./main
```

## Acknowledgements
### Models
The models are retrieved from:

- mobilenetv2-1.0.onnx
	- direct link(may not work in the future): https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.onnx
	- URL: https://github.com/onnx/models/tree/master/vision/classification/mobilenet
	- URL(version specified): https://github.com/onnx/models/tree/8d50e3f598e6d5c67c7c7253e5a203a26e731a1b/vision/classification/mobilenet
- mobilenet_v2_1.0_224_frozen.pb
	- direct link(may not work in the future): https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz
	- URL: https://github.com/tensorflow/models/tree/master/research/slim

