#include <cmath>
#include <numeric>
#include <ostream>

#include "tensorflow/lite/lite_runtime.h"
#include "tensorflow/lite/util.h"

#define OUT_SEQ 100
//#define ODROID_XU4

using namespace cv;
using namespace std;

#ifndef ODROID_XU4
#define RUNTIME_SOCK "/home/nvidia/TfLite_apps/sock/runtime_1"
#define SCHEDULER_SOCK "/home/nvidia/TfLite_apps/sock/scheduler"
#define ROOT_DIR "/home/nvidia/TfLite_apps/image"
#endif

#ifdef ODROID_XU4
#define RUNTIME_SOCK "/home/odroid/TfLite_apps/sock/runtime_1"
#define SCHEDULER_SOCK "/home/odroid/TfLite_apps/sock/scheduler"
#define ROOT_DIR "/home/odroid/TfLite_apps/image"
#endif

#define elapsed_time

std::vector<std::string> coco_label;
std::vector<std::string> imagenet_label;

int ReverseInt(int i) {
  unsigned char ch1, ch2, ch3, ch4;
  ch1 = i & 255;
  ch2 = (i >> 8) & 255;
  ch3 = (i >> 16) & 255;
  ch4 = (i >> 24) & 255;
  return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}
/*

*/
void read_Mnist(string filename, vector<cv::Mat>& vec) {
  ifstream file(filename, ios::binary);
  if (file.is_open()) {
    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = ReverseInt(magic_number);
    file.read((char*)&number_of_images, sizeof(number_of_images));
    number_of_images = ReverseInt(number_of_images);
    file.read((char*)&n_rows, sizeof(n_rows));
    n_rows = ReverseInt(n_rows);
    file.read((char*)&n_cols, sizeof(n_cols));
    n_cols = ReverseInt(n_cols);
    for (int i = 0; i < OUT_SEQ; ++i) {
      cv::Mat tp = Mat::zeros(n_rows, n_cols, CV_8UC1);
      for (int r = 0; r < n_rows; ++r) {
        for (int c = 0; c < n_cols; ++c) {
          unsigned char temp = 0;
          file.read((char*)&temp, sizeof(temp));
          tp.at<uchar>(r, c) = (int)temp;
          // cout << (float)tp.at<uchar>(r, c) << "\n";
        }
      }
      vec.push_back(tp);
      cout << "Get " << i << " Images"
           << "\n";
    }
  } else {
    cout << "file open failed" << endl;
  }
}

void read_Mnist_Label(string filename, vector<unsigned char>& arr) {
  ifstream file(filename, ios::binary);
  if (file.is_open()) {
    for (int i = 0; i < OUT_SEQ; ++i) {
      unsigned char temp = 0;
      file.read((char*)&temp, sizeof(temp));
      if (i > 7) {
        arr.push_back((unsigned char)temp);
      }
    }
  } else {
    cout << "file open failed" << endl;
  }
}

void read_image_opencv(string filename, vector<cv::Mat>& input,
                       tflite::INPUT_TYPE type) {
  cv::Mat cvimg = cv::imread(filename, cv::IMREAD_COLOR);
  if (cvimg.data == NULL) {
    std::cout << "=== IMAGE DATA NULL ===\n";
    return;
  }
  cv::cvtColor(cvimg, cvimg, COLOR_BGR2RGB);
  cv::Mat cvimg_;
  switch (type) {
    case tflite::INPUT_TYPE::IMAGENET224:
      cv::resize(cvimg, cvimg_, cv::Size(224, 224));  // resize
      break;
    case tflite::INPUT_TYPE::IMAGENET256:
      cv::resize(cvimg, cvimg_, cv::Size(256, 256));  // resize
      break;
    case tflite::INPUT_TYPE::IMAGENET300:
      cv::resize(cvimg, cvimg_, cv::Size(300, 300));  // resize
      break;
    case tflite::INPUT_TYPE::IMAGENET416:
      cv::resize(cvimg, cvimg_, cv::Size(416, 416));  // resize
      break;
    case tflite::INPUT_TYPE::COCO416:
      cv::resize(cvimg, cvimg_, cv::Size(416, 416));  // resize
      break;
    case tflite::INPUT_TYPE::LANENET_FRAME:
      cv::resize(cvimg, cvimg_, cv::Size(256, 512));  // resize
      break;
    default:
      break;
  }

  // cvimg_.convertTo(cvimg_, CV_32F, 1.0 / 255.0);
  input.push_back(cvimg_);

  // size should be 224 224 for imagenet and mobilenet
  // size should be 416 416 for yolov4, yolov4_tiny
  // size should be 300 300 for ssd-mobilenetv2-lite
}

void read_image_opencv_quant(string filename, vector<cv::Mat>& input,
                             tflite::INPUT_TYPE type) {
  cv::Mat cvimg = cv::imread(filename, cv::IMREAD_COLOR);
  if (cvimg.data == NULL) {
    std::cout << "=== IMAGE DATA NULL ===\n";
    return;
  }
  cv::cvtColor(cvimg, cvimg, COLOR_BGR2RGB);
  cv::Mat cvimg_;
  switch (type) {
    case tflite::INPUT_TYPE::IMAGENET224:
      cv::resize(cvimg, cvimg_, cv::Size(224, 224));  // resize
      break;
    case tflite::INPUT_TYPE::IMAGENET256:
      cv::resize(cvimg, cvimg_, cv::Size(256, 256));  // resize
      break;
    case tflite::INPUT_TYPE::IMAGENET300:
      cv::resize(cvimg, cvimg_, cv::Size(300, 300));  // resize
      break;

    case tflite::INPUT_TYPE::IMAGENET416:
      cv::resize(cvimg, cvimg_, cv::Size(416, 416));  // resize
      break;
    case tflite::INPUT_TYPE::COCO416:
      cv::resize(cvimg, cvimg_, cv::Size(416, 416));  // resize
      break;
    case tflite::INPUT_TYPE::LANENET_FRAME:
      cv::resize(cvimg, cvimg_, cv::Size(256, 512));  // resize
      break;
    default:
      break;
  }
  cv::Mat quantized;
  cv::Mat converted;

  // Convert the image to 8-bit unsigned integer
  cvimg_.convertTo(converted, CV_8U);

  // Perform quantization using cv::normalize
  cv::normalize(converted, quantized, 0, 255, cv::NORM_MINMAX, CV_8U);

  input.push_back(quantized);
  // size should be 224 224 for imagenet and mobilenet
  // size should be 416 416 for yolov4, yolov4_tiny
  // size should be 300 300 for ssd-mobilenetv2-lite
}

void softmax(std::vector<float>& input, std::vector<float>& output) {
  float maxElement = *std::max_element(input.begin(), input.end());
  float sum = 0.0;
  for (auto const& i : input) sum += std::exp(i - maxElement);
  for (int i = 0; i < input.size(); ++i) {
    output.push_back(std::exp(input[i] - maxElement) / sum);
  }
}

void softmax(std::vector<uint8_t>& input, std::vector<float>& output) {
  uint8_t maxElement = *std::max_element(input.begin(), input.end());
  float sum = 0;
  for (auto const& i : input) sum += std::exp(i - maxElement);
  for (int i = 0; i < input.size(); ++i) {
    output.push_back(std::exp(input[i] - maxElement) / sum);
  }
}

void softmax(std::vector<float>& input, std::vector<float>& output, int begin) {
  input.erase(input.begin(), input.begin() + begin);
  float maxElement = *std::max_element(input.begin(), input.end());
  float sum = 0.0;
  for (auto const& i : input) sum += std::exp(i - maxElement);
  for (int i = 0; i < input.size(); ++i) {
    output.push_back(std::exp(input[i] - maxElement) / sum);
  }
}

// Output parser
void PrintRawOutput(std::vector<std::vector<float>*>* output) {
  for (int i = 0; i < output->size(); ++i) {
    printf("CH [%d]\n", i);
    for (int j = 0; j < output->at(i)->size(); ++j) {
      printf("%.6f \n", output->at(i)->at(j));
    }
    printf("\n");
  }
}

void PrintRawOutputMinMax(std::vector<std::vector<float>*>* output) {
  int max_element =
      std::max_element(output->at(0)->begin(), output->at(0)->end()) -
      output->at(0)->begin();
  printf("[Detection result] \n");
  printf("%d %s, %.6f \n", max_element, imagenet_label[max_element].c_str(),
         output->at(0)->at(max_element));
}

void PrintRawOutputMinMax(std::vector<std::vector<uint8_t>*>* output) {
  int max_element =
      std::max_element(output->at(0)->begin(), output->at(0)->end()) -
      output->at(0)->begin();
  printf("[Detection result] \n");
  printf("%d %s, %d \n", max_element, imagenet_label[max_element].c_str(),
         output->at(0)->at(max_element));
}

float sigmoid(float x) { return 1.0 / (1.0 + std::exp(-x)); }

void ParseOutput(std::vector<std::vector<float>*>* output) {
  std::vector<float> parsed_output;
  if (output->size() ==
      1) {  // Case of single channel output. (usually classification model)
    for (int i = 0; i < output->size(); ++i) {
      softmax(*(output->at(i)), parsed_output);
      int max_element =
          std::max_element(parsed_output.begin(), parsed_output.end()) -
          parsed_output.begin();
      printf("[Detection result] \n");
      printf("%d %s, %.6f \n", max_element, imagenet_label[max_element].c_str(),
             parsed_output[max_element]);
      parsed_output.clear();
    }
    return;
  }
  std::cout << "Got " << output->size() << " outputs to parse"
            << "\n";
  for (int i = 0; i < output->size();
       ++i) {  // Case of multiple channel output. (which contains bbox, obj
               // score, classification score)
    parsed_output.push_back(sigmoid(output->at(i)->at(4)));
    softmax(*(output->at(i)), parsed_output, 5);
    int max_element =
        std::max_element(parsed_output.begin() + 1, parsed_output.end()) -
        parsed_output.begin();
    printf("[Detection result] \n");
    printf("Oscore %.6f, %s %d, %.6f \n", parsed_output[0],
           coco_label[max_element].c_str(), max_element,
           parsed_output[max_element]);
    parsed_output.clear();
  }
  // std::cout << "parsed_outputs : " << parsed_output.size() << "\n";
  // for(int idx=0; idx<parsed_output.size()-1; ++idx){
  // 	printf("%s :  %.6f\n", imagenet_label[idx].c_str(), parsed_output[idx]);
  // }
}

void ParseOutput(std::vector<std::vector<uint8_t>*>* output) {
  if (output->size() ==
      1) {  // Case of single channel output. (usually classification model)
    std::vector<float> parsed_output;
    softmax(*(output->at(0)), parsed_output);
    int max_element =
        std::max_element(output->at(0)->begin(), output->at(0)->end()) -
        output->at(0)->begin();
    printf("[Detection result] \n");
    printf("%d %s, %.6f \n", max_element, imagenet_label[max_element].c_str(),
           parsed_output[max_element]);
    return;
  }
}

void ParseLabels() {
  std::string coco_file = "labels/coco_label.txt";
  std::string imagenet_file = "labels/imagenet_label.txt";
  std::ifstream coco_fd, imagenet_fd;
  coco_fd.open(coco_file);
  imagenet_fd.open(imagenet_file);
  std::string label;
  while (getline(coco_fd, label)) {
    coco_label.push_back(label);
  }
  while (getline(imagenet_fd, label)) {
    imagenet_label.push_back(label);
  }
  std::cout << "Got coco labels : " << coco_label.size() << "\n";
  std::cout << "Got imagenet labels : " << imagenet_label.size() << "\n";
}

tflite::INPUT_TYPE GetInputTypeFromString(string input_type) {
  if (strcmp(input_type.c_str(), "IMAGENET224") == 0) {
    return tflite::INPUT_TYPE::IMAGENET224;
  } else if (strcmp(input_type.c_str(), "IMAGENET256") == 0) {
    return tflite::INPUT_TYPE::IMAGENET256;
  } else if (strcmp(input_type.c_str(), "IMAGENET300") == 0) {
    return tflite::INPUT_TYPE::IMAGENET300;
  } else if (strcmp(input_type.c_str(), "COCO416") == 0) {
    return tflite::INPUT_TYPE::COCO416;
  } else if (strcmp(input_type.c_str(), "MNIST") == 0) {
    return tflite::INPUT_TYPE::MNIST;
  } else if (strcmp(input_type.c_str(), "LANENET") == 0) {
    return tflite::INPUT_TYPE::LANENET_FRAME;
  } else {
    return tflite::INPUT_TYPE::USER;
  }
}

int main(int argc, char* argv[]) {
  const char* first_model;
  const char* second_model;
  std::string input_type_str, sequence_name, log_path;
  bool bUseTwoModel = false;
  bool latency_predictor = false;
  if (argc == 2) {
    std::cout << "Got One Model \n";
    first_model = argv[1];
  } else if (argc == 3) {
    std::cout << "Got Two Model \n";
    bUseTwoModel = true;
    first_model = argv[1];
    second_model = argv[2];
  } else if (argc > 5) {
    std::cout << "Got Two Model and log setups, input type\n";
    bUseTwoModel = true;
    first_model = argv[1];
    second_model = argv[2];
    input_type_str = argv[3];
    sequence_name = argv[4];
    log_path = argv[5];
    latency_predictor = std::stoi(argv[6]);
  } else {
    fprintf(stderr,
            "<tflite model> <tflite model> <input_type> <sequence_name> "
            "<log_path>\n");
    fprintf(stderr, "input_type : IMAGENET224, IMAGENET256, IMAGENET300, COCO416, LANENET\n");
    return 1;
  }

  vector<cv::Mat> input_mnist;
  vector<cv::Mat> input_imagenet;
  vector<cv::Mat> input_iamgenet_quant;
  vector<unsigned char> arr;
  tflite::INPUT_TYPE input_type;
  input_type = GetInputTypeFromString(input_type_str);

// FOR MNIST TEST INPUT
// std::cout << "Loading images \n";
// read_Mnist("images/mnist/train-images-idx3-ubyte", input_mnist);
// std::cout << "Loading Labels \n";
// read_Mnist_Label("labels/mnist/train-labels-idx1-ubyte", arr);
// std::cout << "Loading Mnist Image, Label Complete \n";

#ifndef ODROID_XU4
  // read_image_opencv("/home/nvidia/TfLite_apps/images/lane/lane.jpg",
  //                   input_imagenet, input_type);
  read_image_opencv("/home/nvidia/TfLite_apps/images/imagenet/banana.jpg",
                    input_imagenet, input_type);
  read_image_opencv("/home/nvidia/TfLite_apps/images/imagenet/orange.jpg",
                    input_imagenet, input_type);
  // read_image_opencv("/home/nvidia/TfLite_apps/images/coco/keyboard.jpg",
  // input_imagenet, input_type);
  // read_image_opencv("/home/nvidia/TfLite_apps/images/coco/desk.jpg",
  // input_imagenet, input_type);
  // read_image_opencv_quant("/home/nvidia/TfLite_apps/images/coco/banana_0.jpg",
  // input_iamgenet_quant, input_type);
  // read_image_opencv_quant("/home/nvidia/TfLite_apps/images/coco/orange.jpg",
  // input_iamgenet_quant, input_type);
#endif
#ifdef ODROID_XU4
  read_image_opencv("/home/odroid/TfLite_apps/images/lane/lane.jpg",
                     input_imagenet, input_type);
  read_image_opencv("/home/odroid/TfLite_apps/images/coco/orange.jpg",
                    input_imagenet, input_type);
  //read_image_opencv("/home/odroid/TfLite_apps/images/coco/banana_0.jpg",
  //                  input_imagenet, input_type);
#endif


  tflite::DEVICE_TYPE device_type;
#ifndef ODROID_XU4
  device_type = tflite::DEVICE_TYPE::XAVIER;
#endif
#ifdef ODROID_XU4
  device_type = tflite::DEVICE_TYPE::ODROID;
#endif

  std::vector<double> response_time;
  struct timespec app_begin, inference_begin, inference_end;
  double elapsed_time_ = 0;
  int n = 0;
  std::cout << "Initialize runtime"
            << "\n";
  // Inittialize runtime
  tflite::TfLiteRuntime runtime(RUNTIME_SOCK, SCHEDULER_SOCK, first_model,
                                second_model, input_type, device_type,
                                latency_predictor);
  runtime.SetDeviceType(device_type);
  if (runtime.GetRuntimeState() != tflite::RuntimeState::INVOKE_) {
    std::cout << "Runtime intialization failed"
              << "\n";
    runtime.ShutdownScheduler();
    return 0;
  }
  // Set input type for interpreter
  runtime.SetTestSequenceName(sequence_name);
  runtime.SetLogPath(log_path);
  runtime.InitLogFile();
  runtime.WriteInitStateLog();

  // Output vector
  std::vector<std::vector<float>*>* output;
  std::vector<std::vector<uint8_t>*>* uintoutput;
  ParseLabels();
  std::cout << "Inference start"
            << "\n";
  #ifdef elapsed_time
    clock_gettime(CLOCK_MONOTONIC, &app_begin);
  #endif
  while (n < OUT_SEQ) {
    //std::cout << "[LiteRuntime] invoke : " << n << "\n";
    //runtime.CopyInputToInterpreter(first_model, input_mnist[n % 2],
    //                             input_mnist[n % 2]);
    runtime.CopyInputToInterpreter(first_model, input_imagenet[n % 2],
                                input_imagenet[n % 2]);
    clock_gettime(CLOCK_MONOTONIC, &inference_begin);
    if (runtime.Invoke() != kTfLiteOk) {
      std::cout << "Invoke ERROR"
                << "\n";
      runtime.ShutdownScheduler();
      exit(-1);
    }

    clock_gettime(CLOCK_MONOTONIC, &inference_end);
    if (n >= 0) {  // drop first invoke's data.
      double temp_time = (inference_end.tv_sec - inference_begin.tv_sec) +
                         ((inference_end.tv_nsec - inference_begin.tv_nsec) / 1000000000.0);
  #ifdef elapsed_time
      elapsed_time_ += temp_time;
      printf("%d elapsed_time %.6f latency %.6f", elapsed_time_, temp_time);
  #endif
  #ifndef elapsed_time
      printf("%d latency %.6f \n", n, temp_time);
  #endif
      response_time.push_back(temp_time);
    }
    n++;
    // std::cout << "\n";
    // output = runtime.GetFloatOutputInVector();
    // uintoutput = runtime.GetUintOutputInVector();
    // PrintRawOutputMinMax(output);
    // PrintRawOutputMinMax(uintoutput);
    // PrintRawOutput(output);
    // ParseOutput(uintoutput);
    // ParseOutput(output);
    // std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  runtime.ShutdownScheduler();
  double average_latency = 0;
  for(auto t : response_time){
    average_latency += t;
  }
  average_latency = average_latency / response_time.size();
  printf("Average response time for %d invokes : %.6fs \n", OUT_SEQ, average_latency);
}
