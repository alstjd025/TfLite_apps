# Main experiment rapper
import sys, os
import subprocess
import argparse
from datetime import datetime

now = datetime.now()

exp_name = "Delegation_test"
exp_desc = "gpu_delegation_only_all_combinations"
exp_date = now.date()
exp_time = now.time()
exp_date_time_string = now.strftime('%Y_%m_%d_%Hh_%Mm_%Ss')

def main():
  print("Experiment")
  print(exp_name)
  print(exp_desc)
  print(exp_date_time_string)
  scheduler = subprocess.run()
  
if __name__ == "__main__":
  main()
  
  
