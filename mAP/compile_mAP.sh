#!/bin/bash

AppPath="/home/nvidia/TfLite_apps"
TflitePath="../../FBF-TF/tensorflow/lite/tools/make"
Tensorflowpath="home/nvidia/FBF-TF"

cd ../../FBF-TF 
sudo bazel build -s -c opt --copt="-DMESA_EGL_NO_X11_HEADERS" tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so

sudo bazel build -c opt --define="tflite_with_xnnpack=true" tensorflow/lite/delegates/xnnpack:libxnnpack_delegate.so


cd ../TfLite_apps/mAP
sudo ldconfig
export DISPLAY=:0
echo "TfLite mAP evaluation"

. ${TflitePath}/build_aarch64_lib.sh
touch get_mAP_app.cc
make

