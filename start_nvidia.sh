#!/bin/bash

AppPath="/home/nvidia/TfLite_apps"
TflitePath="../FBF-TF/tensorflow/lite/tools/make"
Tensorflowpath="home/nvidia/FBF-TF"


echo "TfLite test application build"

. ${TflitePath}/build_aarch64_lib.sh
touch TfLite_app.cc
make

