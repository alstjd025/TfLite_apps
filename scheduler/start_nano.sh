#!/bin/bash

AppPath="/home/nano/TfLite_apps"
TflitePath="../FBF-TF/tensorflow/lite/tools/make"
Tensorflowpath="home/nano/FBF-TF"


echo "TfLite test application scheduler build"

. ${TflitePath}/build_aarch64_lib.sh
touch scheduler.cc
make scheduler_nano

