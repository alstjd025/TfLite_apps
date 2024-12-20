#include "tensorflow/lite/tf_scheduler.h"
#include <stdlib.h> 
#include <cstdlib>

// Note: target board config
#define nx
// #define nano
// #define ODROID

#ifdef nx
#define SCHEDULER_SOCK "/home/nvidia/TfLite_apps/sock/scheduler_1"
#define SCHEDULER_ENGINE "/home/nvidia/TfLite_apps/sock/scheduler_e"
#define SCHEDULER_SOCK_2 "/home/nvidia/TfLite_apps/sock/scheduler_2"
#define PARTITIONING_PARAMS \
  "/home/nvidia/TfLite_apps/params/subgraph/[model_type]/"
#endif

#ifdef nano
#define SCHEDULER_SOCK "/home/nano/TfLite_apps/sock/scheduler_1"
#define SCHEDULER_ENGINE "/home/nano/TfLite_apps/sock/scheduler_e"
#define SCHEDULER_SOCK_2 "/home/nano/TfLite_apps/sock/scheduler_2"
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
  if (argc < 2) {
    std::cout
        << "ERROR on argument. Usage : ./scheduler [recovery_on/off(1, 0)] partitioning_param_1 partitioning_param_2 ..."
        << "\n";
    exit(-1);
  }
  // Parse parameters
  int params = argc;
  std::vector<std::string> param_file_names;
  for(int i=2; i<params; ++i){
    param_file_names.push_back(argv[i]);
  }
  bool recovery = false;
  if(atoi(argv[1]) == 0){
    recovery = false;
  }else{
    recovery = true;
  }
  tflite::TfScheduler scheduler(SCHEDULER_SOCK, SCHEDULER_SOCK_2, SCHEDULER_ENGINE, param_file_names);
  scheduler.Work(recovery);

  return 0;
}
