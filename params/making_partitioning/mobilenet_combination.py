from itertools import product

TF_P_PLAN_CPU = 0
TF_P_PLAN_GPU = 1
TF_P_PLAN_CO_E = 2
TF_P_PLAN_CPU_XNN = 3
TF_P_PLAN_CO_E_XNN = 4
file_ = ["cpu", "gpu", "co_e", "xnn", "co_e_xnn"]
plan_ratio_cw = [2, 4, 5, 6, 8]
plan_ratio_hw = [12, 14, 15, 16, 18] #CW/HW ratio

class delegation_combination:
    def mobilenet_combination(self, resource, f_name):
        file_name = "../model/mobilenet/mobilenet_combination_" + f_name
        m = open(file_name, 'a')
        # layer num
        layer = 31

        # change to layer by test case
        plan_resource = [TF_P_PLAN_CPU, resource] #resource type
        plan_idx = 6 # subgraph num

        idx = 0
        # change to layer by models
        num = [0] * layer
        name = [0] * layer
        with open('mobilenet_layer', mode = 'r+', encoding='UTF-8') as l:    
            while True:
                lines = l.readline()
                if not lines:
                    break
                num[idx], name[idx] = lines.split(' ')
                
        # per subgraph's usable resource set
        # repeat = fallback num + 1(subgraph in no fallback layer)
        if f_name == 'cpu' or f_name == 'gpu':
            nREr = list(product(plan_resource, repeat=plan_idx))
        else:
            nREr = list(product(plan_resource, repeat=plan_idx+1))
        nCWr = list(product(plan_ratio_cw, repeat=plan_idx))
        nHWr = list(product(plan_ratio_cw, repeat=plan_idx))
        # need to change model file
        for j in range(len(nREr)):
            count = 0 # for checking resource type in combination(nREr)
            k = 0
            while k < layer:
                if k < 6: # condition has to change by model structure
                    m.write('{0}\n'.format(k))
                    while True:
                        if k==6: # condition has to change by model structure)
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
                elif k < 18: # condition has to change by model structure
                    m.write('{0}\n'.format(k))
                    while True:
                        if k==18: # condition has to change by model structure)
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
                elif k < 27: # condition has to change by model structure
                    m.write('{0}\n'.format(k))
                    while True:
                        if k==27: # condition has to change by model structure)
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
                else:
                    m.write('{0}\n'.format(k))
                    k = 31
                    m.write('{0}\n'.format(k))
                    if f_name == 'cpu' or f_name == 'gpu':
                        m.write('{0}\n'.format(0))
                    else:
                        m.write('{0}\n'.format(nREr[j][count]))
                    m.write('{0}\n'.format(0))
                    m.write('{0}\n'.format(-1))
                    m.write('{0}\n'.format(-2))
        m.close()
class co_execution_combination:
    def mobilenet_combination(self, resource, f_name):
            file_name = "../model/mobilenet/mobilenet_combination_" + f_name 
            m = open(file_name, 'a')
            # layer num
            layer = 31

            # change to layer by test case
            plan_resource = [TF_P_PLAN_CPU, TF_P_PLAN_GPU, TF_P_PLAN_CPU_XNN] #resource type
            plan_last = [TF_P_PLAN_CPU, TF_P_PLAN_CPU_XNN]
            plan_idx = 6 # subgraph num

            idx = 0
            # change to layer by models
            num = [0] * layer
            name = [0] * layer
            with open('mobilenet_layer', mode = 'r+', encoding='UTF-8') as r:    
                while True:
                    lines = r.readline()
                    if not lines:
                        break
                    num[idx], name[idx] = lines.split(' ')
            # per subgraph's usable resource set
            # repeat = fallback num + 1(subgraph in no fallback layer)
            nREr = list(product(plan_last, repeat=1)) # resource product
            nREr_1 = list(product(plan_resource, repeat=1)) # resource product
            nCWr = list(product(plan_ratio_cw, repeat=2)) # cw product
            nHWr = list(product(plan_ratio_hw, repeat=plan_idx-1)) # hw product
            # need to change model file
            for i in range(len(nREr)):
                for j in range(len(nHWr)): # len(nHWr) == len(nCWr)
                    for w in range(len(nREr_1)): # len(nHWr) == len(nCWr)
                        counth = 0 # for checking ratio in combination(nCWr)
                        countc = 0
                        countr = 0
                        k = 0
                        while k < layer:
                            if k < 6: # condition has to change by model structure
                                m.write('{0}\n'.format(k))
                                while True:
                                    if k==6: # condition has to change by model structure)
                                        break
                                    else:
                                        k += 1
                                m.write('{0}\n'.format(k))
                                m.write('{0}\n'.format(resource))
                                m.write('{0}\n'.format(nHWr[j][counth]))
                                counth += 1
                            elif k < 12: # condition has to change by model structure
                                m.write('{0}\n'.format(k))
                                while True:
                                    if k==12: # condition has to change by model structure)
                                        break
                                    else:
                                        k += 1
                                m.write('{0}\n'.format(k))
                                m.write('{0}\n'.format(resource))
                                m.write('{0}\n'.format(nHWr[j][counth]))
                                counth += 1
                            elif k < 18: # condition has to change by model structure
                                m.write('{0}\n'.format(k))
                                while True:
                                    if k==18: # condition has to change by model structure)
                                        break
                                    else:
                                        k += 1
                                m.write('{0}\n'.format(k))
                                m.write('{0}\n'.format(resource))
                                m.write('{0}\n'.format(nHWr[j][counth]))
                                counth += 1
                            elif k < 24: # condition has to change by model structure
                                m.write('{0}\n'.format(k))
                                while True:
                                    if k==24: # condition has to change by model structure)
                                        break
                                    else:
                                        k += 1
                                m.write('{0}\n'.format(k))
                                m.write('{0}\n'.format(resource))
                                m.write('{0}\n'.format(nHWr[j][counth]))
                                counth += 1
                            elif k < 27: # condition has to change by model structure
                                m.write('{0}\n'.format(k))
                                while True:
                                    if k==27: # condition has to change by model structure)
                                        break
                                    else:
                                        k += 1
                                m.write('{0}\n'.format(k))
                                m.write('{0}\n'.format(resource))
                                m.write('{0}\n'.format(nHWr[j][counth]))
                                counth += 1
                            elif k < 29: # condition has to change by model structure
                                m.write('{0}\n'.format(k))
                                while True:
                                    if k==29: # condition has to change by model structure)
                                        break
                                    else:
                                        k += 1
                                m.write('{0}\n'.format(k))
                                m.write('{0}\n'.format(nREr_1[w][countc]))
                                m.write('{0}\n'.format(0))
                                countc += 1
                            else:
                                m.write('{0}\n'.format(k))
                                k = 31
                                m.write('{0}\n'.format(k))
                                if f_name == 'cpu' or f_name == 'gpu':
                                    m.write('{0}\n'.format(0))
                                else:
                                    m.write('{0}\n'.format(nREr[i][countr]))
                                    countr += 1
                                m.write('{0}\n'.format(0))
                                m.write('{0}\n'.format(-1))
                                m.write('{0}\n'.format(-2))
            m.close()

def main():
    # delegate = delegation_combination()
    # for i in range(len(file_)):
    #     if(i==0 or i==1 or i==3): #just make file for delegation
    #         delegate.mobilenet_combination(i, file_[i])
    co_e = co_execution_combination()
    for i in range(len(file_)):
        if(i==2 or i==4): #just make file for delegation
            co_e.mobilenet_combination(i, file_[i])
main()