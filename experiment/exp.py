# Main experiment rapper
import sys, os
import subprocess
import argparse
import time
from datetime import datetime
from threading import Thread
from threading import Lock

now = datetime.now()
h_dir = '/home/nvidia/TfLite_apps'
scheduler_dir = h_dir + '/scheduler/scheduler'
app_dir = h_dir + '/TfLite_app'
models = ['yolov4-tiny-416_ieie.tflite', 
          'efficientnet_lite4_fp32_2.tflite',
          'mobilenet_v1_10_224_fp32.tflite']


yolo_params = h_dir + '/params/model/yolo'
efficientnet_params = h_dir + '/params/model/efficient'
mobilenet_params = h_dir + '/params/model/mobilenet'
generated_params = h_dir + '/params/generated_sequences'
param_folders = [yolo_params, efficientnet_params, mobilenet_params]
model_and_params = {'yolov4-tiny-416_ieie.tflite' : yolo_params,
                    'efficientnet_lite4_fp32_2.tflite' : efficientnet_params,
                    'mobilenet_v1_10_224_fp32.tflite' : mobilenet_params}

exp_name = "Delegation_test"
exp_desc = "gpu_delegation_only_all_combinations"
exp_date = now.date()
exp_time = now.time()
exp_date_time_string = now.strftime('%Y_%m_%d_%Hh_%Mm_%Ss')
global_sequence = 1
seq_lock = Lock()


def generate_exp_sequences(param_folder_path):
  print("Generates exp sequences for all partitioning params")
  sequence = 1
  output_file_path = param_folder_path + '/sequences'
  try:
    # get the file list of folder
    file_list = os.listdir(param_folder_path)
    for file_name in file_list:
      file_path = os.path.join(param_folder_path, file_name)
      if os.path.isfile(file_path): 
        output_file_name = os.path.join(output_file_path, str(sequence))
        print(f"Reading contents of {file_name}:")
        with open(file_path, 'r') as file:
          f = open(output_file_name, 'w')
          for line in file:
            stripped_line = line.strip()
            f.write(stripped_line + '\n')
            if(stripped_line == '-2'):
              f.close()
              sequence += 1
              output_file_name = os.path.join(output_file_path, str(sequence))
              f = open(output_file_name, 'w')
        print("=" * 40)  # line seperate
  except Exception as e:
      print(f"An error occurred: {e}")
  
def generate_exp_sequences_(param_folder_path):
  print("Generates exp sequences for all partitioning params")
  # os.rmdir(param_folder_path, 'sequences')
  os.mkdir(os.path.join(param_folder_path, 'sequences'))
  base_output_file_path = param_folder_path + '/sequences'
  try:
    # get the file list of folder
    file_list = os.listdir(param_folder_path)
    for file_name in file_list:
      sequence = 1
      file_path = os.path.join(param_folder_path, file_name)
      output_file_path = base_output_file_path
      if os.path.isfile(file_path):       
        # os.rmdir(output_file_path, file_name)
        os.mkdir(os.path.join(output_file_path, file_name))
        output_file_path = os.path.join(output_file_path, file_name)
        output_file_name = os.path.join(output_file_path, str(sequence))
        print("output file name : ", output_file_name)
        print(f"Reading contents of {file_name}")
        with open(file_path, 'r') as file:
          f = open(output_file_name, 'w')
          for line in file:
            stripped_line = line.strip()
            if(stripped_line == '-2'):
              f.close()
              sequence += 1
              output_file_name = os.path.join(output_file_path, str(sequence))
              f = open(output_file_name, 'w')
          os.remove(output_file_name)
        print("=" * 40)  # line seperate
  except Exception as e:
    print(f"An error occurred: {e}")
    
  
  
def scheduler_thread(param):
  print("Scheduler thread")
  scheduler = subprocess.run([scheduler_dir, param])
  
  
def runtime_thread(model):
  print("Runtime thread")  
  runtime = subprocess.run([app_dir, model])
  
  
def main():
  print("Experiment")
  print(exp_name)
  print(exp_desc)
  print(exp_date_time_string)
  for folder in param_folders:
    generate_exp_sequences_(folder)
  for model, param in model_and_params.items():
    # Get param file
    file_list = os.listdir(os.path.join(param, 'sequences'))
    for file_name in file_list:
      sequence_list = os.listdir(file_name)
      for sequence in sequence_list:
        sched_thd = Thread(target=scheduler_thread, args=(param))
        time.sleep(1)
        runt_thd = Thread(target=runtime_thread, args=(model))
    
      
  # for model in models:
  #   for sequence in global_sequence:
    
  
if __name__ == "__main__":
  main()
  
  
