#!/bin/bash

SchedulerPath="/home/odroid/TfLite_apps/scheduler"
TflitePath="../../FBF-TF/tensorflow/lite/tools/make"
Tensorflowpath="home/odroid/FBF-TF"


echo "TfLite scheduler Test"

. ${TflitePath}/build_bbb_lib.sh

cd ${SchedulerPath}
touch scheduler.cc
make scheduler_odroid

