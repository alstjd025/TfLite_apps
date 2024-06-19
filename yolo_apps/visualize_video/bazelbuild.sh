cd ../../../FBF-TF 
sudo bazel build -s -c opt --copt="-DMESA_EGL_NO_X11_HEADERS" tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so

sudo bazel build -c opt --define="tflite_with_xnnpack=true" tensorflow/lite/delegates/xnnpack:libxnnpack_delegate.so

cd ../TfLite_apps/yolo_apps/visualize_video
