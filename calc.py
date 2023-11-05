for i in range(1):
    # file_path = '/home/nvidia/TfLite_apps/all/thread6/yolo_combination_ivs_param_' + str(i) + '_seq_' + str(i) +'_timestamps.txt'
    file_path = '/home/nvidia/TfLite_apps/1_timestamps.txt'
    file_s_path = '/home/nvidia/TfLite_apps/1_s_timestamps.txt'
    
    # 평균을 계산할 변수 및 카운터 초기화
    total = 0.0
    value = 0.0
    count = 0
    
    s_total = 0.0
    s_value = 0.0
    s_count = 0
    # 파일 열기 및 처리
    
    with open(file_path, 'r') as f:
       for line in f:
            if line == 'LOG_START\n':
                f.readline()
                while True:
                    line = f.readline().strip()
                    if line == '': break
                    value = float(line)
                    total += value
                    count += 1
                    
    with open(file_s_path, 'r') as s:
       for line in s:
            if line == 'LOG_START\n':
                s.readline()
                while True:
                    line = s.readline().strip()
                    if line == '': break
                    s_value = float(line)
                    s_total += s_value
                    s_count += 1
                    
    # 평균 계산
    if count > 0:
        average = total / count
        
        print(f'{average:.9f}')
    if s_count > 0:
        s_average = s_total / s_count
        print(f'{s_average:.9f}')
    else:
        print('유효한 값이 없습니다.')