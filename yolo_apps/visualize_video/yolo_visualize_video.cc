#include "tensorflow/lite/lite_runtime.h"
#include "tensorflow/lite/util.h"
#include <cmath>
#include <numeric>
#include <ostream>

using namespace cv;
using namespace std;

#define RUNTIME_SOCK "/home/nvidia/TfLite_apps/sock/runtime_1"
#define SCHEDULER_SOCK "/home/nvidia/TfLite_apps/sock/scheduler"

#define REAL_FPS 30

void SaveData4mAP(std::vector<tflite::YOLO_Parser::BoundingBox>& result_boxes, int fnum, std::map<int, std::string>& labelDict) {
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

void visualize_with_labels(cv::Mat& image, const std::vector<tflite::YOLO_Parser::BoundingBox>& bboxes, std::map<int, std::string>& labelDict, float fps) {
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
    
		// std::string fps_label = "FPS: " + std::to_string(fps);
    	// cv::putText(image, fps_label, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
	}
}

bool STOP_SIGNAL(cv::Mat& image,std::vector<tflite::YOLO_Parser::BoundingBox>& result_boxes,std::map<int, std::string>& labelDict, int frame_num) {
	bool SIGNAL = false;
	for (int i=0; i <result_boxes.size(); i++) { 
		auto object_name = labelDict[result_boxes[i].class_id];
		// auto left = result_boxes[i].left;
		// auto top = result_boxes[i].top;
		// auto right = result_boxes[i].right;
		// auto bottom = result_boxes[i].bottom;
		// auto cls_data = result_boxes[i].score;
		if(object_name == "person") SIGNAL = true;
	}
	if(SIGNAL) {
		std::string label = "STOP SIGNAL";
    	cv::putText(image, label, cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
		// std::string fn = "Frame num : " + to_string(frame_num);
    	// cv::putText(image, fn, cv::Point(10, 70), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 1);
		return true;
	}
	return false;
  }

int main(int argc, char* argv[]) {
	const char* first_model;
	const char* second_model;
	std::string input_type_str, sequence_name, log_path, input_type_;
	bool bUseTwoModel = false;
	bool latency_predictor = false;
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
		// std::cout << "Got Two Model and log setups, input type\n";
		bUseTwoModel = true;
		first_model = argv[1];
		second_model = argv[2];
		input_type_str = argv[3];
		sequence_name = argv[4];
		log_path = argv[5];
		latency_predictor = std::stoi(argv[6]);
	}
	else{
			fprintf(stderr, "minimal <tflite model>\n");
			return 1;
	}
	tflite::DEVICE_TYPE device_type;
	#ifndef ODROID_
	device_type = tflite::DEVICE_TYPE::XAVIER;
	#endif
	#ifdef ODROID_
	device_type = tflite::DEVICE_TYPE::ODROID;
	#endif
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
	// std::cout << "Initialize runtime" << "\n";
	// Inittialize runtime
	tflite::INPUT_TYPE input_type;
	input_type = tflite::INPUT_TYPE::COCO416;
    tflite::TfLiteRuntime runtime(RUNTIME_SOCK, SCHEDULER_SOCK,
    							first_model, second_model, input_type, device_type, latency_predictor);
	runtime.SetDeviceType(device_type);
	if(runtime.GetRuntimeState() != tflite::RuntimeState::INVOKE_){
		std::cout << "Runtime intialization failed" << "\n";
		runtime.ShutdownScheduler();
		return 0;
	}
	runtime.SetTestSequenceName(sequence_name);
	runtime.SetLogPath(log_path);
	runtime.InitLogFile();
	runtime.WriteInitStateLog();

	// For video capture
	std::string video_filename = "../yolo_apps/visualize_video/40.mp4";
	// std::string video_filename = "../yolo_apps/visualize_video/EAI.webm";
	cv::VideoCapture video_capture(video_filename);
    if (!video_capture.isOpened()) {
        std::cerr << "Error: Could not open video file: " << video_filename << std::endl;
        return -1;
    }
	int frame_count=0;
	float fps = 0.0;
	int frame_num=1;
	cv::Mat video_frame;
	struct timespec begin, end;
	struct timespec Begin, End;
	////////////////////////////////////////////////////////////
	// TODO point
	// Make demo app synchronous with real-time stamp (30 fps) (0.033s)
	////////////////////////////////////////////////////////////
	int frame_index =0;
	int next_frame=0;
	// <Real World time stamp>  -----> <Ours time stamp>
	// <Real World time stamp>  <--REAL_FPS--- <Ours time stamp>
	// choosen video capture index should be [Ours time stamp / Real World time stamp]
	////////////////////////////////////////////////////////////
	double total_time = 0;
	int detected_frame = 0;
	double fps_sum=0;
	bool SIG_END = false;
	std::string output_folder = "output_frames/"; ///// 0625
	clock_gettime(CLOCK_MONOTONIC, &Begin);
	while (video_capture.read(video_frame)) {
		////////////////////////////////////////////
		if(frame_index != next_frame) {
			frame_index+=1;
			// std::cout << "JUMP FRAME\n";
			continue;
		}
		frame_index = 0;
		frame_num+=next_frame;
		////////////////////////////////////////////
		clock_gettime(CLOCK_MONOTONIC, &begin);
        input.clear();
		cv::cvtColor(video_frame, video_frame, COLOR_BGR2RGB); 
		cv::resize(video_frame, video_frame, cv::Size(416,416));
		input.push_back(video_frame);
		runtime.CopyInputToInterpreter(first_model, input[0], input[0]);
        if(runtime.Invoke() != kTfLiteOk){
      		std::cout << "Invoke ERROR" << "\n";
			runtime.ShutdownScheduler();
      		exit(-1);
    	}
		clock_gettime(CLOCK_MONOTONIC, &end);
		std::vector<tflite::YOLO_Parser::BoundingBox> bboxes = tflite::YOLO_Parser::result_boxes;
		
		// for mAP
		// SaveData4mAP(bboxes, fnum, labelDict);
		
		// visualize
		std::string window_name = "parsed image";
		cv::namedWindow(window_name, cv::WINDOW_NORMAL);
		double temp_time = (end.tv_sec - begin.tv_sec) +
                         ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
		fps = 1/temp_time;
		fps_sum+=fps;
		detected_frame+=1;
		// printf("<<<<<<<<<<<<<<<<<<<<<<< FPS : %f >>>>>>>>>>>>>>>>>>>>>>>>>>>\n", fps);
        // printf("<<<<<<<<<<<<<<<<<<<<<< Frame number : %d >>>>>>>>>>>>>>>>>>>>\n", frame_num);
    	if (!video_frame.empty()) visualize_with_labels(video_frame, bboxes, labelDict, fps);
		else std::cerr << "Error: Unable to load the image: " << video_filename << std::endl;
		////////////////////////////////////////////////////////
		total_time +=temp_time;
		double sync_param = 1.0/REAL_FPS;
		sync_param = static_cast<double>(std::round((sync_param)*1000)/1000);
		next_frame = temp_time / sync_param; // to sync with real-world time stamp
		// printf("<<<<<<<<<<<<<<<<<<<< Next frame number is :%d >>>>>>>>>>>>>>>>>>>>>>\n", next_frame);
		bool signal = STOP_SIGNAL(video_frame,bboxes,labelDict, frame_num);

		if (signal && !SIG_END) {
			clock_gettime(CLOCK_MONOTONIC, &End);
			double end_time = (End.tv_sec - Begin.tv_sec) +
                         ((End.tv_nsec - Begin.tv_nsec) / 1000000000.0);
			// std::string time = "Total time : " + to_string(end_time) + "s";
    		// cv::putText(video_frame, time, cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 1);
			
			// cv::imshow(window_name, video_frame);
		    printf("\033[0;31m<<<<<<<<< STOP SIGNAL >>>>>>>>>\033[0m\n");
			printf("..................... Response time        is : %.3fs    .....................\n", end_time);
			printf("..................... Human detected frame is : %d       .....................\n", frame_num);
			printf("..................... Total detected frame is : %d/%d    .....................\n", detected_frame, frame_num);
			printf("..................... Average inference fps is : %.3ffps .....................\n", fps_sum/detected_frame);
			printf("..................... Last inference latency is : %.3fms .....................\n", temp_time*1000);
			std::string output_image = output_folder + "result_image.png"; ///// 0625
            cv::imwrite(output_image, video_frame);
			std::string output_log = output_folder + "result_log.txt"; 
			std::ofstream outFile(output_log);
			outFile << "Response time = " << end_time << "s" << std::endl;
			outFile << "Human detected frame = " << frame_num << std::endl;
			outFile << "Total detected frame = " << detected_frame << "/" <<  frame_num<< std::endl;
			outFile << "Average inference fps = " << fps_sum/detected_frame << "fps" << std::endl;
			outFile << "Last inference latency = " << temp_time*1000 << "ms" << std::endl;
			outFile.close();
			// cv::waitKey(0);
			SIG_END = true;
		}
	    else{
			// cv::imshow(window_name, video_frame);
			std::string output_image = output_folder + "frame_" + std::to_string(frame_num) + ".png";
            cv::imwrite(output_image, video_frame);
			// cv::waitKey(1);
		}	
        frame_num+=1;
		////////////////////////////////////////////////////////
    }
	runtime.ShutdownScheduler();
    // cv::waitKey(0);
	// cv::destroyAllWindows();
}
