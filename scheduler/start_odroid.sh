#!/bin/bash

UnitSimple="/home/odroid/TfLite_apps"
TflitePath="../../FBF-TF/tensorflow/lite/tools/make"
Tensorflowpath="home/odroid/FBF-TF"


echo "TfLite scheduler Test"

. ${TflitePath}/build_bbb_lib.sh
touch scheduler.cc
make scheduler_odroid

