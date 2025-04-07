#include "tensorflow/lite/tf_scheduler.h"
#include <stdlib.h> 
#include <cstdlib>

// Note: target board config
#define nx
// #define nano
// #define ODROID

#ifdef nx
// signature 1
#define SCHEDULER_SOCK__1 "/home/nvidia/TfLite_apps/sock_1/scheduler_1"
#define SCHEDULER_ENGINE__1 "/home/nvidia/TfLite_apps/sock_1/scheduler_e"
#define SCHEDULER_SOCK_2__1 "/home/nvidia/TfLite_apps/sock_1/scheduler_2"

// signature 2
#define SCHEDULER_SOCK__2 "/home/nvidia/TfLite_apps/sock_2/scheduler_1"
#define SCHEDULER_ENGINE__2 "/home/nvidia/TfLite_apps/sock_2/scheduler_e"
#define SCHEDULER_SOCK_2__2 "/home/nvidia/TfLite_apps/sock_2/scheduler_2"

// signature 3
#define SCHEDULER_SOCK__3 "/home/nvidia/TfLite_apps/sock_3/scheduler_1"
#define SCHEDULER_ENGINE__3 "/home/nvidia/TfLite_apps/sock_3/scheduler_e"
#define SCHEDULER_SOCK_2__3 "/home/nvidia/TfLite_apps/sock_3/scheduler_2"

#define PARTITIONING_PARAMS \
  "/home/nvidia/TfLite_apps/params/subgraph/[model_type]/"
#endif

#ifdef nano
// signature 1
#define SCHEDULER_SOCK "/home/nano/TfLite_apps/sock_1/scheduler_1"
#define SCHEDULER_ENGINE "/home/nano/TfLite_apps/sock_1/scheduler_e"
#define SCHEDULER_SOCK_2 "/home/nano/TfLite_apps/sock_1/scheduler_2"

// signature 2
#define SCHEDULER_SOCK "/home/nano/TfLite_apps/sock_2/scheduler_1"
#define SCHEDULER_ENGINE "/home/nano/TfLite_apps/sock_2/scheduler_e"
#define SCHEDULER_SOCK_2 "/home/nano/TfLite_apps/sock_2/scheduler_2"

// signature 3
#define SCHEDULER_SOCK "/home/nano/TfLite_apps/sock_3/scheduler_1"
#define SCHEDULER_ENGINE "/home/nano/TfLite_apps/sock_3/scheduler_e"
#define SCHEDULER_SOCK_2 "/home/nano/TfLite_apps/sock_3/scheduler_2"

#define PARTITIONING_PARAMS \
  "/home/nano/TfLite_apps/params/subgraph/[model_type]/"
#endif

#ifdef ODROID
#define SCHEDULER_SOCK "/home/odroid/TfLite_apps/sock/scheduler_1"
#define SCHEDULER_ENGINE "/home/odroid/TfLite_apps/sock/scheduler_e"
#define SCHEDULER_SOCK_2 "/home/odroid/TfLite_apps/sock/scheduler_2"
#define PARTITIONING_PARAMS \
  "/home/odroid/TfLite_apps/params/subgraph/[model_type]/"
#endif

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout
        << "ERROR on argument. Usage : ./scheduler [recovery_on/off(1, 0)] [socket signature(1,2,,)] partitioning_param_1 partitioning_param_2 ..."
        << "\n";
    exit(-1);
  }
  // Parse parameters
  int params = argc;
  int socket_signature = 1;
  char* scheduler_sock;
  char* scheduler_sock_2;
  char* scheduler_engine;
  std::vector<std::string> param_file_names;
  for(int i=3; i<params; ++i){
    param_file_names.push_back(argv[i]);
  }
  bool recovery = false;
  if(atoi(argv[1]) == 0){
    std::cout << "[Scheduler] recovery OFF" << "\n";
    recovery = false;
  }else{
    std::cout << "[Scheduler] recovery ON" << "\n";
    recovery = true;
  }
  socket_signature = atoi(argv[2]);
  std::cout << "[Scheduler] socker signature " << socket_signature << "\n";
  switch (socket_signature)
  {
  case 1:
    scheduler_sock = SCHEDULER_SOCK__1;
    scheduler_sock_2 = SCHEDULER_SOCK_2__1;
    scheduler_engine = SCHEDULER_ENGINE__1;
    break;
  case 2:
    scheduler_sock = SCHEDULER_SOCK__2;
    scheduler_sock_2 = SCHEDULER_SOCK_2__2;
    scheduler_engine = SCHEDULER_ENGINE__2;
    break;
  case 3:
    scheduler_sock = SCHEDULER_SOCK__3;
    scheduler_sock_2 = SCHEDULER_SOCK_2__3;
    scheduler_engine = SCHEDULER_ENGINE__3;
    break;
  default:
    scheduler_sock = SCHEDULER_SOCK__1; 
    scheduler_sock_2 = SCHEDULER_SOCK_2__1;
    scheduler_engine = SCHEDULER_ENGINE__1;
    break;
  }


  tflite::TfScheduler scheduler(scheduler_sock, scheduler_sock_2, scheduler_engine, param_file_names);
  scheduler.Work(recovery);

  return 0;
}
