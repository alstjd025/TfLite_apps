#include "tensorflow/lite/lite_runtime.h"
#include "tensorflow/lite/util.h"
#include <cmath>
#include <numeric>
#include <ostream>

#define OUT_SEQ 300
#define lanenet
#define twomodel

using namespace cv;
using namespace std;

#define RUNTIME_SOCK "/home/nvidia/TfLite_apps/sock/runtime_1"
#define SCHEDULER_SOCK "/home/nvidia/TfLite_apps/sock/scheduler"

std::vector<std::string> coco_label;
std::vector<std::string> imagenet_label;

void read_image_opencv(string filename, vector<cv::Mat>& input){
	cv::Mat cvimg = cv::imread(filename, cv::IMREAD_COLOR);
	if(cvimg.data == NULL){
		std::cout << "=== IMAGE DATA NULL ===\n";
		return;
	}
	cv::cvtColor(cvimg, cvimg, COLOR_BGR2RGB); 
	cv::Mat cvimg_;
	cv::resize(cvimg, cvimg_, cv::Size(416,416));
	input.push_back(cvimg_);
}

void YOLO_parsing(std::vector<tflite::YOLO_Parser::BoundingBox>& result_boxes, int fnum, std::map<int, std::string>& labelDict) {
	std::string filename = "../../mAP_TF/input/detection-results/" + std::to_string(fnum) + ".txt";
	std::ofstream outFile(filename);
	if (!outFile.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }
	for (int i=0; i <result_boxes.size(); i++) { 
		auto object_name = labelDict[result_boxes[i].class_id];
		auto left = result_boxes[i].left;
		auto top = result_boxes[i].top;
		auto right = result_boxes[i].right;
		auto bottom = result_boxes[i].bottom;
		auto cls_data = result_boxes[i].score;
		outFile << object_name << " " <<  cls_data << " ";
		outFile << left << " " << top << " " << right << " " << bottom; 
		outFile << std::endl;
	}
	outFile.close();
  }

void visualize_with_labels(cv::Mat& image, const std::vector<tflite::YOLO_Parser::BoundingBox>& bboxes, std::map<int, std::string>& labelDict) {
    for (const tflite::YOLO_Parser::BoundingBox& bbox : bboxes) {
        int x1 = bbox.left;
        int y1 = bbox.top;
        int x2 = bbox.right;
        int y2 = bbox.bottom;
        cv::RNG rng(bbox.class_id);
        cv::Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        int label_x = x1;
        int label_y = y1 - 20;

        cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), color, 3);
        std::string object_name = labelDict[bbox.class_id];
        float confidence_score = bbox.score;
        std::string label = object_name + ": " + std::to_string(confidence_score);
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr);
        cv::rectangle(image, cv::Point(x1, label_y - text_size.height), cv::Point(x1 + text_size.width, label_y + 5), color, -1);
        cv::putText(image, label, cv::Point(x1, label_y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    }
}

tflite::INPUT_TYPE GetInputTypeFromString(string input_type){
  if(strcmp(input_type.c_str(), "IMAGENET224") == 0){
    return tflite::INPUT_TYPE::IMAGENET224;
  }else if(strcmp(input_type.c_str(), "IMAGENET300") == 0){
    return tflite::INPUT_TYPE::IMAGENET300;
  }else if(strcmp(input_type.c_str(), "COCO416") == 0){
    return tflite::INPUT_TYPE::COCO416;
  }else if(strcmp(input_type.c_str(), "MNIST") == 0){
    return tflite::INPUT_TYPE::MNIST;
  }else{
    return tflite::INPUT_TYPE::USER;
  }
}

int main(int argc, char* argv[]) {
	const char* first_model;
	const char* second_model;
	std::string input_type_str, sequence_name, log_path;
	bool bUseTwoModel = false;
	if (argc == 2) {
		std::cout << "Got One Model \n";
		first_model = argv[1];
	}else if(argc == 3){
		std::cout << "Got Two Model \n";
		bUseTwoModel = true;
		first_model = argv[1];
		second_model = argv[2];
	}
	else if(argc > 5){
		std::cout << "Got Two Model and log setups, input type\n";
		bUseTwoModel = true;
		first_model = argv[1];
		second_model = argv[2];
		input_type_str = argv[3];
		sequence_name = argv[4];
		log_path = argv[5];
	}
	else{
			fprintf(stderr, "minimal <tflite model>\n");
			return 1;
	}
    vector<cv::Mat> input;	    
    std::map<int, std::string> labelDict = {
        {0, "person"},     {1, "bicycle"},   {2, "car"},          {3, "motorbike"},
        {4, "aeroplane"},  {5, "bus"},       {6, "train"},        {7, "truck"},
        {8, "boat"},       {9, "traffic_light"}, {10, "fire_hydrant"}, {11, "stop_sign"},
        {12, "parking_meter"}, {13, "bench"}, {14, "bird"},       {15, "cat"},
        {16, "dog"},       {17, "horse"},    {18, "sheep"},       {19, "cow"},
        {20, "elephant"},  {21, "bear"},     {22, "zebra"},       {23, "giraffe"},
        {24, "backpack"},  {25, "umbrella"}, {26, "handbag"},     {27, "tie"},
        {28, "suitcase"},  {29, "frisbee"},  {30, "skis"},        {31, "snowboard"},
        {32, "sports_ball"}, {33, "kite"},   {34, "baseball_bat"}, {35, "baseball_glove"},
        {36, "skateboard"}, {37, "surfboard"}, {38, "tennis_racket"}, {39, "bottle"},
        {40, "wine_glass"}, {41, "cup"},     {42, "fork"},        {43, "knife"},
        {44, "spoon"},     {45, "bowl"},    {46, "banana"},      {47, "apple"},
        {48, "sandwich"},  {49, "orange"},  {50, "broccoli"},    {51, "carrot"},
        {52, "hot_dog"},   {53, "pizza"},   {54, "donut"},       {55, "cake"},
        {56, "chair"},     {57, "sofa"},    {58, "potted_plant"}, {59, "bed"},
        {60, "dining_table"}, {61, "toilet"}, {62, "tvmonitor"}, {63, "laptop"},
        {64, "mouse"},     {65, "remote"},  {66, "keyboard"},    {67, "cell_phone"},
        {68, "microwave"}, {69, "oven"},    {70, "toaster"},     {71, "sink"},
        {72, "refrigerator"}, {73, "book"}, {74, "clock"},       {75, "vase"},
        {76, "scissors"},  {77, "teddy_bear"}, {78, "hair_drier"}, {79, "toothbrush"}
    };
    int fnum = 0;
	std::cout << "Initialize runtime" << "\n";
	// Inittialize runtime
	tflite::INPUT_TYPE input_type;
	input_type_str = "COCO416"; //HOONING
	input_type = GetInputTypeFromString(input_type_str);
    tflite::TfLiteRuntime runtime(RUNTIME_SOCK, SCHEDULER_SOCK,
    							first_model, second_model, input_type);
	if(runtime.GetRuntimeState() != tflite::RuntimeState::INVOKE_){
		std::cout << "Runtime intialization failed" << "\n";
		runtime.ShutdownScheduler();
		return 0;
	}
	runtime.SetTestSequenceName(sequence_name);
	runtime.SetLogPath(log_path);
	runtime.InitLogFile();
	runtime.WriteInitStateLog();

    while(fnum < OUT_SEQ){
		fnum+=1;
		input.clear();
		std::string filename = "../../mAP_TF/input/images-optional/" + std::to_string(fnum) + ".jpg";
		read_image_opencv(filename, input);
		runtime.CopyInputToInterpreter(first_model, input[0], input[0]);
        if(runtime.Invoke() != kTfLiteOk){
      		std::cout << "Invoke ERROR" << "\n";
			runtime.ShutdownScheduler();
      		exit(-1);
    	}
		std::cout << filename << std::endl;
		std::vector<tflite::YOLO_Parser::BoundingBox> bboxes = tflite::YOLO_Parser::result_boxes;
		YOLO_parsing(bboxes, fnum, labelDict);
		
		// visualize
		// std::string window_name = std::to_string(fnum) + "'s parsed image";
		// cv::namedWindow(window_name, cv::WINDOW_NORMAL);
		// cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);
		// cv::resize(image, image, cv::Size(416,416)); 
    	// if (!image.empty()) {
    	    // visualize_with_labels(image, bboxes, labelDict);
        	// cv::imshow(window_name, image);
		// }
		// else {
    		// std::cerr << "Error: Unable to load the image: " << filename << std::endl;
		// }
    }
	runtime.ShutdownScheduler();
    // cv::waitKey(0);
	// cv::destroyAllWindows();
}
