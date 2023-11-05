from itertools import product

TF_P_PLAN_CPU = 0
TF_P_PLAN_GPU = 1
TF_P_PLAN_CO_E = 2
TF_P_PLAN_CPU_XNN = 3
TF_P_PLAN_CO_E_XNN = 4
file_dl = ["cpu", "gpu", 0, "xnn"]
file_co = [0, 0, "co_e", 0, "co_e_xnn"]
plan_ratio_cw = [2, 4, 5, 6, 8]
plan_ratio_hw = [13, 14, 15, 16, 17] #CW/HW ratio

class delegation_combination:
    def yolo_combination(self, resource, f_name):
        file_name = "../model/yolo/yolo_combination_ivs" + f_name
        m = open(file_name, 'a')
        # layer num
        layer = 31

        
        plan_idx = 6 # subgraph num

        idx = 0
        # change to layer by models
        num = [0] * layer
        name = [0] * layer
        # with open('yolo_layer', mode = 'r+', encoding='UTF-8') as l:    
        #     while True:
        #         lines = l.readline()
        #         if not lines:
        #             break
        #         num[idx], name[idx] = lines.split(' ')
                
        # per subgraph's usable resource set
        # repeat = fallback num + 1(subgraph in no fallback layer)
        if f_name == 'cpu':
            # change to layer by test case
            plan_resource = [TF_P_PLAN_CPU] #resource type
            nREr = list(product(plan_resource, repeat=plan_idx))
            print(nREr)
        else:
            # change to layer by test case
            plan_resource = [0, resource] #resource type
            nREr = list(product(plan_resource, repeat=plan_idx))
            print(nREr)
        # need to change model file
        for j in range(len(nREr)):
            if f_name == 'gpu' or f_name == 'xnn':
               if j==0 : continue 
            count = 0 # for checking resource type in combination(nREr)
            k = 0
            while k < layer:
                if k < 4: # condition has to change by model structure
                    m.write('{0}\n'.format(k))
                    while True:
                        if k==4: # condition has to change by model structure)
                            break
                        else:
                            k += 1
                    m.write('{0}\n'.format(k))
                    m.write('{0}\n'.format(nREr[j][count]))
                    count += 1
                    m.write('{0}\n'.format(0))
                elif k < 8: # condition has to change by model structure
                    m.write('{0}\n'.format(k))
                    while True:
                        if k==8: # condition has to change by model structure)
                            break
                        else:
                            k += 1
                    m.write('{0}\n'.format(k))
                    m.write('{0}\n'.format(nREr[j][count]))
                    count += 1
                    m.write('{0}\n'.format(0))
                elif k < 12: # condition has to change by model structure
                    m.write('{0}\n'.format(k))
                    while True:
                        if k==12: # condition has to change by model structure)
                            break
                        else:
                            k += 1
                    m.write('{0}\n'.format(k))
                    m.write('{0}\n'.format(nREr[j][count]))
                    count += 1
                    m.write('{0}\n'.format(0))
                elif k < 24: # condition has to change by model structure
                    m.write('{0}\n'.format(k))
                    while True:
                        if k==24: # condition has to change by model structure)
                            break
                        else:
                            k += 1
                    m.write('{0}\n'.format(k))
                    m.write('{0}\n'.format(nREr[j][count]))
                    count += 1
                    m.write('{0}\n'.format(0))
                elif k < 29: # condition has to change by model structure
                    m.write('{0}\n'.format(k))
                    while True:
                        if k==29: # condition has to change by model structure)
                            break
                        else:
                            k += 1
                    m.write('{0}\n'.format(k))
                    m.write('{0}\n'.format(nREr[j][count]))
                    count += 1
                    m.write('{0}\n'.format(0))
                elif k < 30: # condition has to change by model structure
                    m.write('{0}\n'.format(k))
                    while True:
                        if k==30: # condition has to change by model structure)
                            break
                        else:
                            k += 1
                    m.write('{0}\n'.format(k))
                    m.write('{0}\n'.format(0))
                    m.write('{0}\n'.format(0))
                elif k < 31:
                    m.write('{0}\n'.format(k))
                    while True:
                        if k==31: # condition has to change by model structure)
                            break
                        else:
                            k += 1
                    m.write('{0}\n'.format(k))
                    m.write('{0}\n'.format(nREr[j][count]))
                    count += 1
                    m.write('{0}\n'.format(0))
                    m.write('{0}\n'.format(-1))
                    m.write('{0}\n'.format(-2))
        m.close()
class co_execution_combination:
    def yolo_combination(self):
        file_name = "../model/yolo/yolo_combination_co_exe"
        f = open(file_name, 'a')
        # layer num
        layer = 31

        # change to layer by test case
        plan_resource = [TF_P_PLAN_CO_E, TF_P_PLAN_CO_E_XNN] #resource type
        
        nREr = list(product(plan_resource, repeat=1)) # resource product
        nHWr = list(product(plan_ratio_hw, repeat=1)) # hw product
        print(nREr)
        print(nHWr)
        for i in range(len(nREr)): # len(nHWr) == len(nCWr)
            for j in range(len(nHWr)): # len(nHWr) == len(nCWr)
                k = 24
                while k < layer: # CW
                    if k <29: # fallback subgraph
                        f.write('{0}\n'.format(k))
                        while True:
                            if k == 29: break
                            else: k += 1
                        f.write('{0}\n'.format(k))
                        f.write('{0}\n'.format(nREr[i][0]))
                        f.write('{0}\n'.format(nHWr[j][0]))
                        f.write('{0}\n'.format(-1))
                        f.write('{0}\n'.format(-2))
                        break
        f.close()
class acc_combination():
    def acc_yolo_combination(self):
        file_name = "../model/yolo/yolo_combination_ivs"
        m = open(file_name, 'a')
        # layer num
        layer = 118
        plan_idx = 6 # subgraph num
        idx = 0
        # change to layer by models
        num = [0] * layer
        name = [0] * layer
        # with open('yolo_layer', mode = 'r+', encoding='UTF-8') as l:
        #     while True:
        #         lines = l.readline()
        #         if not lines:
        #             break
        #         num[idx], name[idx] = lines.split(' ')
                
        # per subgraph's usable resource set
        # need to change model file
        plan_resource = [2, 4] #resource type
        hw_resource = [15, 13] #resource type
        nREr = list(product(plan_resource, repeat=1))
        nHWr = list(product(hw_resource, repeat=1))
        print(nREr)
        print(nHWr)
        for j in range(len(nREr)):
            count = 0 # for checking resource type in combination(nREr)
            k = 0
            if k < 8: # condition has to change by model structure
                m.write('{0}\n'.format(k))
                while True:
                    if k==8: # condition has to change by model structure)
                        break
                    else:
                        k += 1
                m.write('{0}\n'.format(k))
                m.write('{0}\n'.format(nREr[j][0]))
                m.write('{0}\n'.format(13))
                m.write('{0}\n'.format(-1))
                m.write('{0}\n'.format(-2))
                # elif k < 8: # condition has to change by model structure
                #     m.write('{0}\n'.format(k))
                #     while True:
                #         if k==8: # condition has to change by model structure)
                #             break
                #         else:
                #             k += 1
                #     m.write('{0}\n'.format(k))
                #     m.write('{0}\n'.format(nREr[j][count]))
                #     count += 1
                #     m.write('{0}\n'.format(0))
                # elif k < 12: # condition has to change by model structure
                #     m.write('{0}\n'.format(k))
                #     while True:
                #         if k==12: # condition has to change by model structure)
                #             break
                #         else:
                #             k += 1
                #     m.write('{0}\n'.format(k))
                #     m.write('{0}\n'.format(nREr[j][count]))
                #     count += 1
                #     m.write('{0}\n'.format(0))
                # elif k < 24: # condition has to change by model structure
                #     m.write('{0}\n'.format(k))
                #     while True:
                #         if k==24: # condition has to change by model structure)
                #             break
                #         else:
                #             k += 1
                #     m.write('{0}\n'.format(k))
                #     m.write('{0}\n'.format(nREr[j][count]))
                #     count += 1
                #     m.write('{0}\n'.format(0))
                # elif k < 29: # condition has to change by model structure
                #     m.write('{0}\n'.format(k))
                #     while True:
                #         if k==29: # condition has to change by model structure)
                #             break
                #         else:
                #             k += 1
                #     m.write('{0}\n'.format(k))
                #     m.write('{0}\n'.format(nREr[j][count]))
                #     count += 1
                #     m.write('{0}\n'.format(0))
                # elif k < 30: # condition has to change by model structure
                #     m.write('{0}\n'.format(k))
                #     while True:
                #         if k==30: # condition has to change by model structure)
                #             break
                #         else:
                #             k += 1
                #     m.write('{0}\n'.format(k))
                #     m.write('{0}\n'.format(0))
                #     m.write('{0}\n'.format(0))
                # elif k < 31:
                #     m.write('{0}\n'.format(k))
                #     while True:
                #         if k==31: # condition has to change by model structure)
                #             break
                #         else:
                #             k += 1
                #     m.write('{0}\n'.format(k))
                #     m.write('{0}\n'.format(nREr[j][count]))
                #     count += 1
                #     m.write('{0}\n'.format(0))
                #     m.write('{0}\n'.format(-1))
                #     m.write('{0}\n'.format(-2))
        m.close()
def main():
    # delegate = delegation_combination()
    # for i in range(len(file_dl)):
    #     if i==2: continue
    #     delegate.yolo_combination(i, file_dl[i])
    # co_e = co_execution_combination()
    # co_e.yolo_combination()
    acc = acc_combination()
    acc.acc_yolo_combination()
main()
