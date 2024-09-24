#!/bin/bash

AppPath="/home/odroid/TfLite_apps"
TflitePath="../FBF-TF/tensorflow/lite/tools/make"
Tensorflowpath="home/odroid/FBF-TF"


echo "TfLite test application build"

. ${TflitePath}/build_bbb_lib.sh

cd ${AppPath}
touch TfLite_app.cc
make TfLite_app_odroid

