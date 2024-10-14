#include "tensorflow/lite/tf_scheduler.h"
//#define ODROID

#ifndef ODROID
#define SCHEDULER_SOCK "/home/nvidia/TfLite_apps/sock/scheduler_1"
#define SCHEDULER_SOCK_2 "/home/nvidia/TfLite_apps/sock/scheduler_2"
#define PARTITIONING_PARAMS \
  "/home/nvidia/TfLite_apps/params/subgraph/[model_type]/"
#endif

#ifdef ODROID
#define SCHEDULER_SOCK "/home/odroid/TfLite_apps/sock/scheduler_1"
#define SCHEDULER_SOCK_2 "/home/odroid/TfLite_apps/sock/scheduler_2"
#define PARTITIONING_PARAMS \
  "/home/odroid/TfLite_apps/params/subgraph/[model_type]/"
#endif

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cout
        << "ERROR on argument. Usage : ./scheduler partitioning_param_1 partitioning_param_2 ..."
        << "\n";
    exit(-1);
  }
  // Parse parameters
  int params = argc;
  std::vector<std::string> param_file_names;
  for(int i=1; i<params; ++i){
    param_file_names.push_back(argv[i]);
  }

  tflite::TfScheduler scheduler(SCHEDULER_SOCK, SCHEDULER_SOCK_2, param_file_names);
  scheduler.Work();

  return 0;
}
