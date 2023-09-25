#!/bin/bash

AppPath="/home/odroid/TfLite_apps"
TflitePath="../FBF-TF/tensorflow/lite/tools/make"
Tensorflowpath="home/odroid/FBF-TF"


echo "TfLite Unit_simple Test"

. ${TflitePath}/build_bbb_lib.sh.sh
touch TfLite_app.cc
make TfLite_app_odroid

