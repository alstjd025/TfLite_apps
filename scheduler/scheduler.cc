#include "tensorflow/lite/tf_scheduler.h"

#define SCHEDULER_SOCK "/home/nvidia/TfLite_apps/sock/scheduler"
#define PARTITIONING_PARAMS "/home/nvidia/TfLite_apps/params/subgraph/[model_type]/[partitioning_plan]"

int main(int argc, char* argv[]){
  if(argc < 2){
    std::cout << "ERROR on argument. Usage : ./scheduler <partitioning_param_dir>" << "\n";
    exit(-1);
  }

  tflite::TfScheduler scheduler(SCHEDULER_SOCK, argv[1]);
  scheduler.Work();

  return 0;
}
