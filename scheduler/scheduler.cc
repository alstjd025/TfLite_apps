#include "tensorflow/lite/tf_scheduler.h"

#define SCHEDULER_SOCK "/home/nvidia/TfLite_apps/sock/scheduler"
#define PARTITIONING_PARAMS "/home/nvidia/TfLite_apps/params/subgraph/[model_type]/[partitioning_plan]"

int main(){
  tflite::TfScheduler scheduler(SCHEDULER_SOCK, PARTITIONING_PARAMS);
  scheduler.Work();

  return 0;
}
