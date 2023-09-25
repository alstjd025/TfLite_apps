#include "tensorflow/lite/tf_scheduler.h"
#define ODROID

#ifndef ODROID
  #define SCHEDULER_SOCK "/home/nvidia/TfLite_apps/sock/scheduler"
  #define PARTITIONING_PARAMS "/home/nvidia/TfLite_apps/params/subgraph/[model_type]/"
#endif

#ifdef ODROID
  #define SCHEDULER_SOCK "/home/odroid/TfLite_apps/sock/scheduler"
  #define PARTITIONING_PARAMS "/home/odroid/TfLite_apps/params/subgraph/[model_type]/"
#endif


int main(int argc, char* argv[]){
  if(argc < 2){
    std::cout << "ERROR on argument. Usage : ./scheduler <partitioning_param_dir>" << "\n";
    exit(-1);
  }

  tflite::TfScheduler scheduler(SCHEDULER_SOCK, argv[1]);
  scheduler.Work();

  return 0;
}
