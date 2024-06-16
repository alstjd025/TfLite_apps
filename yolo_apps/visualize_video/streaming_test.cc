#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <time.h>

using namespace std;
using namespace cv;

int main()
{
    // Receiving Pipeline with multiple queues and videorate set to 30 FPS
    //VideoCapture cap("udpsrc port=5000 ! application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96 ! rtph264depay ! queue ! h264parse ! queue ! avdec_h264 ! queue ! videorate ! video/x-raw,framerate=30/1 ! videoconvert ! queue ! appsink", CAP_GSTREAMER);
    
    // Receiving Pipleline with multiple queues (4 times)
    //VideoCapture cap("udpsrc port=5000 ! application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96 ! rtph264depay ! queue ! h264parse ! queue ! avdec_h264 ! queue ! videorate ! video/x-raw,framerate=20/1 ! videoconvert ! queue ! appsink", CAP_GSTREAMER);

    // [BEST]
    //VideoCapture cap("udpsrc port=5000 ! application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96 ! rtph264depay ! h264parse ! queue ! avdec_h264 ! queue! videorate ! video/x-raw,framerate=20/1 ! videoconvert ! appsink", CAP_GSTREAMER);
    
    // Ex
    //VideoCapture cap("udpsrc port=5000 ! application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96 ! rtph264depay ! h264parse ! queue ! avdec_h264 ! queue! videorate ! video/x-raw,framerate=15/1 ! videoconvert ! appsink", CAP_GSTREAMER);
        
    // Old 
    //VideoCapture cap("udpsrc port=5000 ! application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink", CAP_GSTREAMER);
 
    // Experi [min boundary_fps 20/1]
    VideoCapture cap("udpsrc port=5000 ! application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96 ! rtph264depay ! h264parse ! queue ! avdec_h264 ! queue! videorate ! video/x-raw,framerate=20/1 ! videoconvert ! appsink", CAP_GSTREAMER);
    
    // Experi [no limit for framerate]
    //VideoCapture cap("udpsrc port=5000 ! application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96 ! rtph264depay ! h264parse ! queue ! avdec_h264 ! queue! videorate ! video/x-raw! videoconvert ! appsink", CAP_GSTREAMER);


    if (!cap.isOpened()) {
        cerr << "VideoCapture not opened !!!" << endl;
        exit(-1);
    }

    struct timespec begin, end;
    double fps;
    
    // Initialize the begin timestamp
    clock_gettime(CLOCK_MONOTONIC, &begin);

    while (true) {
        Mat frame;
        cap.read(frame);
        if (frame.empty()) {
            cerr << "Empty frame received" << endl;
            break;
        }

        // Get the end timestamp
        clock_gettime(CLOCK_MONOTONIC, &end);
        
        // Calculate the time difference between frames
        double temp_time = (end.tv_sec - begin.tv_sec) +
                           ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
        
        // Calculate FPS
        fps = 1.0 / temp_time;

        // Print FPS
        printf("FPS: %.2f\n", fps);

        // Update the begin timestamp
        begin = end;

        imshow("receiver", frame);

        if (waitKey(1) == 27) {
            break;
        }
    }

    // When everything done, release the video capture object
    cap.release();
    // Closes all the frames
    destroyAllWindows();

    return 0;
}

