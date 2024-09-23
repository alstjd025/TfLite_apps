#include "tensorflow/lite/lite_runtime.h"
#include "tensorflow/lite/util.h"
#include <cmath>
#include <numeric>
#include <ostream>
#include <iostream>
#include <cstring>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>


using namespace cv;
using namespace std;

#define RUNTIME_SOCK "/home/nvidia/TfLite_apps/sock/runtime_1"
#define SCHEDULER_SOCK "/home/nvidia/TfLite_apps/sock/scheduler"

#define REAL_FPS 30
#define PORT 8080
#define SERVER_IP "192.168.0.6"
#define BUFFER_SIZE 1024

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

bool STOP_SIGNAL(cv::Mat& image,std::vector<tflite::YOLO_Parser::BoundingBox>& result_boxes,std::map<int, std::string>& labelDict) {
	bool SIGNAL = false;
	for (int i=0; i <result_boxes.size(); i++) { 
		auto object_name = labelDict[result_boxes[i].class_id];
		if(object_name == "person") SIGNAL = true;
	}
	if(SIGNAL) {
		std::string label = "STOP SIGNAL";
    	cv::putText(image, label, cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
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
	//socket init
	int sockfd;
    struct sockaddr_in server_addr;
    char buffer[BUFFER_SIZE];

    // 소켓 생성
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }

    memset(&server_addr, 0, sizeof(server_addr));

    // 서버 정보 설정
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    server_addr.sin_addr.s_addr = inet_addr(SERVER_IP);
	std::string message = "0";
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
	// Initialize runtime
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
	std::cout << "before capture" << "\n";	
	// For streaming camera frame
	//cv::VideoCapture video_capture("udpsrc port=5000 ! application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96 ! rtph264depay ! h264parse ! queue ! omxh264dec ! queue! videorate ! video/x-raw,framerate=15/1 ! videoconvert ! appsink", CAP_GSTREAMER);
	//cv::VideoCapture video_capture("udpsrc port=5000 ! application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96 ! rtph264depay ! h264parse ! queue ! decodebin ! queue! videorate ! video/x-raw,framerate=15/1 ! nvvideoconvert ! videoconvert ! appsink", CAP_GSTREAMER);
	// cv::VideoCapture video_capture("udpsrc port=5000 ! application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96 ! rtph264depay ! h264parse ! queue ! avdec_h264 ! queue! videorate ! video/x-raw,framerate=10/1 ! videoconvert ! appsink", CAP_GSTREAMER);
	// cv::VideoCapture video_capture("udpsrc port=5000 ! application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96 ! rtph264depay ! h264parse ! queue ! avdec_h264 ! queue! videorate ! video/x-raw,framerate=15/1 ! videoconvert ! appsink", CAP_GSTREAMER);
	//cv::VideoCapture video_capture("udpsrc port=5000 ! application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96 ! rtph264depay ! h264parse ! queue ! avdec_h264 ! queue! videorate ! video/x-raw! videoconvert ! appsink", CAP_GSTREAMER);
	cv::VideoCapture video_capture("udpsrc port=5000 ! application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96 ! rtph264depay ! h264parse ! queue ! avdec_h264 ! queue! videorate ! video/x-raw,framerate=20/1 ! videoconvert ! appsink", CAP_GSTREAMER);
	std::cout << "make video_capture instance\n";
	if (!video_capture.isOpened()) {
        return -1;
    }
	int frame_count=0;
	float fps = 0.0;
	int frame_num=1;
	cv::Mat video_frame;
	struct timespec begin, end;
	struct timespec Begin, End;
	int frame_index =0;
	int next_frame=0;
	double total_time = 0;
	int detected_frame = 0;
	double fps_sum=0;
	std::string window_name = "parsed image";
	cv::namedWindow(window_name, cv::WINDOW_NORMAL);
	double temp_time=0;
	double sync_param=0;
	bool signal = false;
	bool SIG_END = false;
	std::string output_folder = "output_frames/"; 
	std::cout << "connecting to host" << "\n";
	sendto(sockfd, message.c_str(), message.size(), MSG_CONFIRM, (const struct sockaddr *)&server_addr, sizeof(server_addr));
	int n = recvfrom(sockfd, (char *)buffer, BUFFER_SIZE, MSG_WAITALL, nullptr, nullptr);
	buffer[n] = '\0';
	std::cout << "Server: " << buffer << std::endl;

	clock_gettime(CLOCK_MONOTONIC, &Begin);
	while (video_capture.read(video_frame)) {
		if(frame_index != next_frame) {
			frame_index+=1;
			// std::cout << "JUMP FRAME\n";
			continue;
		}
		frame_index = 0;
		frame_num+=next_frame;
		clock_gettime(CLOCK_MONOTONIC, &begin);
        input.clear();
		input.push_back(video_frame);
		runtime.CopyInputToInterpreter(first_model, input[0], input[0]);
        if(runtime.Invoke() != kTfLiteOk){
      		std::cout << "Invoke ERROR" << "\n";
			runtime.ShutdownScheduler();
      		exit(-1);
    	}
		clock_gettime(CLOCK_MONOTONIC, &end);
		std::vector<tflite::YOLO_Parser::BoundingBox> bboxes = tflite::YOLO_Parser::result_boxes;
		temp_time = (end.tv_sec - begin.tv_sec) +
                         ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
		fps = 1/temp_time;
		fps_sum+=fps;
		detected_frame+=1;
    	if (!video_frame.empty()) visualize_with_labels(video_frame, bboxes, labelDict);
		else std::cerr << "Error: Unable to load the image: " <<  std::endl;
		total_time +=temp_time;
		sync_param = 1.0/REAL_FPS;
		sync_param = static_cast<double>(std::round((sync_param)*1000)/1000);
		next_frame = temp_time / sync_param; // to sync with real-world time stamp
		signal = STOP_SIGNAL(video_frame,bboxes,labelDict);
		if (signal && !SIG_END) {
			std::string e_stop_msg = "1";
			sendto(sockfd, e_stop_msg.c_str(), e_stop_msg.size(), MSG_CONFIRM, (const struct sockaddr *)&server_addr, sizeof(server_addr));
			clock_gettime(CLOCK_MONOTONIC, &End);
			double end_time = (End.tv_sec - Begin.tv_sec) +
                         ((End.tv_nsec - Begin.tv_nsec) / 1000000000.0);
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
			//cv::imshow(window_name, video_frame);
			std::string output_image = output_folder + "frame_" + std::to_string(frame_num) + ".png";
            cv::imwrite(output_image, video_frame);
			//cv::waitKey(1);
		}	
        frame_num+=1;
		////////////////////////////////////////////////////////
    }
	runtime.ShutdownScheduler();
    // cv::waitKey(0);
}
