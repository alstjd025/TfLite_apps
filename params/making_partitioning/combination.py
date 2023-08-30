from itertools import product

TF_P_PLAN_CPU = 0
TF_P_PLAN_GPU = 1
TF_P_PLAN_CO_E = 2
TF_P_PLAN_CPU_XNN = 3
TF_P_PLAN_CO_E_XNN = 4
file_ = ["cpu", "gpu", "co_e", "xnn", "co_e_xnn"]
plan_ratio_cw = [1, 2, 3, 4, 5, 6, 7, 8, 9]
plan_ratio_hw = [13, 14, 15, 16, 17] #CW/HW ratio

class delegation_combination:
    def __init__(self, flag, name):
        self.resource = flag
        self.name = name
    def yolo_combination(self):
        file_name = "../model/yolo/yolo_combination_" + self.name
        f = open(file_name, 'w')
        # layer num
        layer = 152

        # change to layer by test case
        plan_resource = [TF_P_PLAN_CPU, self.resource] #resource type
        
        plan_idx = 0 # subgraph num

        idx = 0
        # change to layer by models
        num = [0] * layer
        name = [0] * layer
        with open('yolo_layer', mode = 'r+', encoding='UTF-8') as r:    
            while True:
                lines = r.readline()
                if not lines:
                    break
                num[idx], name[idx] = lines.split(' ')
                if(name[idx] == 'SPLIT\n' or num[idx] == '55'): # condition has to change by model structure
                    plan_idx += 1
                idx += 1
        not_fallback = plan_idx
        # per subgraph's usable resource set
        # repeat = fallback num + 1(subgraph in no fallback layer)
        nREr = list(product(plan_resource, repeat=not_fallback))
        nCWr = list(product(plan_ratio_cw, repeat=not_fallback))
        nHWr = list(product(plan_ratio_cw, repeat=not_fallback))
        # need to change model file
        for j in range(len(nREr)):
            count = 0 # for checking resource type in combination(nREr)
            k = 0
            while k < layer:
                if(name[k] == 'SPLIT\n'): # condition has to change by model structure
                    f.write('{0}\n'.format(k))
                    f.write('{0}\n'.format(k+1))
                    f.write('{0}\n'.format(0))
                    f.write('{0}\n'.format(0))
                    count += 1
                    k += 1
                elif k > 54:
                    f.write('{0}\n'.format(k))
                    while True:
                        if k==152: break
                        k+=1
                    f.write('{0}\n'.format(k))
                    f.write('{0}\n'.format(0))
                    f.write('{0}\n'.format(0))
                    f.write('{0}\n'.format(-1))
                    f.write('{0}\n'.format(-2))
                else:
                    f.write('{0}\n'.format(k))
                    while True:
                        if(num[k] == '55' or name[k] == 'SPLIT\n'): # condition has to change by model structure)
                            break
                        else:
                            k += 1
                    f.write('{0}\n'.format(k))
                    f.write('{0}\n'.format(nREr[j][count]))
                    f.write('{0}\n'.format(0))
        f.close()
        r.close()

    def mobilenet_combination(self):
        file_name = "../model/mobilenet/mobilenet_combination_" + self.name
        m = open(file_name, 'w')
        # layer num
        layer = 31

        # change to layer by test case
        plan_resource = [TF_P_PLAN_CPU, self.resource] #resource type
        plan_idx = 0 # subgraph num

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
                if(name[idx] == 'SQUEEZE\n'): # condition has to change by model structure
                    plan_idx += 1
                idx += 1
        not_fallback = plan_idx+1
        # per subgraph's usable resource set
        # repeat = fallback num + 1(subgraph in no fallback layer)
        nREr = list(product(plan_resource, repeat=not_fallback))
        nCWr = list(product(plan_ratio_cw, repeat=not_fallback))
        nHWr = list(product(plan_ratio_cw, repeat=not_fallback))
        # need to change model file
        for j in range(len(nREr)):
            count = 0 # for checking resource type in combination(nREr)
            k = 0
            while k < layer:
                if(name[k] == 'SQUEEZE\n'): # condition has to change by model structure
                    m.write('{0}\n'.format(k))
                    m.write('{0}\n'.format(k+1))
                    m.write('{0}\n'.format(0))
                    m.write('{0}\n'.format(0))
                    count += 1
                    k += 1
                elif k == 30:
                    m.write('{0}\n'.format(k))
                    m.write('{0}\n'.format(k+1))
                    m.write('{0}\n'.format(0))
                    m.write('{0}\n'.format(0))
                    m.write('{0}\n'.format(-1))
                    m.write('{0}\n'.format(-2))
                    k+=1
                else:
                    m.write('{0}\n'.format(k))
                    while True:
                        if(name[k] == 'SQUEEZE\n'): # condition has to change by model structure)
                            break
                        else:
                            k += 1
                    m.write('{0}\n'.format(k))
                    m.write('{0}\n'.format(nREr[j][count]))
                    m.write('{0}\n'.format(plan_ratio_cw[0]))
        m.close()
        l.close()

    def efficient_combination(self):
        file_name = "../model/efficient/efficient_combination_" + self.name
        m = open(file_name, 'w')
        # layer num
        layer = 118

        # change to layer by test case
        plan_resource = [TF_P_PLAN_CPU, self.resource] #resource type
        plan_idx = 0 # subgraph num
        idx = 0
        # change to layer by models
        num = [0] * layer
        name = [0] * layer
        with open('efficient_layer', mode = 'r+', encoding='UTF-8') as l:    
            while True:
                lines = l.readline()
                if not lines:
                    break
                num[idx], name[idx] = lines.split(' ')
                if(name[idx] == 'RESHAPE\n'): # condition has to change by model structure
                    plan_idx += 1
                idx += 1
        not_fallback = plan_idx+1
        tot_subgraph = plan_idx*2
        # per subgraph's usable resource set
        # repeat = fallback num + 1(subgraph in no fallback layer)
        nREr = list(product(plan_resource, repeat=not_fallback))
        nCWr = list(product(plan_ratio_cw, repeat=not_fallback))
        nHWr = list(product(plan_ratio_cw, repeat=not_fallback))
        # need to change model file
        for j in range(len(nREr)):
            count = 0 # for checking resource type in combination(nREr)
            k = 0
            while k < layer:
                if(name[k] == 'RESHAPE\n'): # condition has to change by model structure
                    m.write('{0}\n'.format(k))
                    m.write('{0}\n'.format(k+1))
                    m.write('{0}\n'.format(0))
                    m.write('{0}\n'.format(0))
                    count += 1
                    k += 1
                elif k == 116:
                    m.write('{0}\n'.format(k))
                    while True:
                        if k==layer:
                            break
                        k+=1
                    m.write('{0}\n'.format(k))
                    m.write('{0}\n'.format(0))
                    m.write('{0}\n'.format(plan_ratio_cw[0]))
                    m.write('{0}\n'.format(-1))
                    m.write('{0}\n'.format(-2))
                    k+=1
                else:
                    m.write('{0}\n'.format(k))
                    while True:
                        if(name[k] == 'RESHAPE\n'): # condition has to change by model structure)
                            break
                        else:
                            k += 1
                    m.write('{0}\n'.format(k))
                    m.write('{0}\n'.format(nREr[j][count]))
                    m.write('{0}\n'.format(plan_ratio_cw[0]))
        m.close()
        l.close()

class co_execution_combination:
    def __init__(self, flag, name):
        self.resource = flag
        self.name = name
    def yolo_combination(self):
        file_name = "../model/yolo/yolo_combination_" + self.name
        f = open(file_name, 'w')
        # layer num
        layer = 152

        # change to layer by test case
        plan_resource = [TF_P_PLAN_CPU, self.resource] #resource type
        
        plan_idx = 0 # subgraph num

        idx = 0
        # change to layer by models
        num = [0] * layer
        name = [0] * layer
        with open('yolo_layer', mode = 'r+', encoding='UTF-8') as r:    
            while True:
                lines = r.readline()
                if not lines:
                    break
                num[idx], name[idx] = lines.split(' ')
                if(name[idx] == 'SPLIT\n' or num[idx] == '55'): # condition has to change by model structure
                    plan_idx += 1
                idx += 1
        not_fallback = plan_idx
        # per subgraph's usable resource set
        # repeat = fallback num + 1(subgraph in no fallback layer)
        nREr = list(product(plan_resource, repeat=not_fallback)) # resource product
        nHWr = list(product(plan_ratio_hw, repeat=not_fallback)) # hw product
        # need to change model file
        for j in range(len(nHWr)): # len(nHWr) == len(nCWr)
            count = 0 # for checking ratio in combination(nCWr)
            k = 0
            while k < layer: # CW
                if(name[k] == 'SPLIT\n'): # fallback subgraph
                    f.write('{0}\n'.format(k))
                    f.write('{0}\n'.format(k+1))
                    f.write('{0}\n'.format(0))
                    f.write('{0}\n'.format(0))
                    count += 1 
                    k += 1
                elif k > 54: # last subgraph 55~152
                    f.write('{0}\n'.format(k))
                    while True:
                        if k==152: break
                        k+=1
                    f.write('{0}\n'.format(k))
                    f.write('{0}\n'.format(0))
                    f.write('{0}\n'.format(0))
                    f.write('{0}\n'.format(-1))
                    f.write('{0}\n'.format(-2))
                else:
                    f.write('{0}\n'.format(k))
                    while True:
                        if(num[k] == '55' or name[k] == 'SPLIT\n'): # condition has to change by model structure)
                            break
                        else:
                            k += 1
                    f.write('{0}\n'.format(k))
                    f.write('{0}\n'.format(self.resource))
                    f.write('{0}\n'.format(nHWr[j][count]))
        f.close()
        r.close()

    def mobilenet_combination(self):
            file_name = "../model/mobilenet/mobilenet_combination_" + self.name
            f = open(file_name, 'w')
            # layer num
            layer = 31

            # change to layer by test case
            plan_resource = [TF_P_PLAN_CPU, self.resource] #resource type
            
            plan_idx = 0 # subgraph num

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
                    if(name[idx] == 'SQUEEZE\n'): # condition has to change by model structure
                        plan_idx += 1
                    idx += 1
            not_fallback = plan_idx + 1
            # per subgraph's usable resource set
            # repeat = fallback num + 1(subgraph in no fallback layer)
            nREr = list(product(plan_resource, repeat=not_fallback)) # resource product
            nCWr = list(product(plan_ratio_cw, repeat=1)) # cw product
            nHWr = list(product(plan_ratio_hw, repeat=1)) # hw product
            nRAr = [nCWr, nHWr]
            # need to change model file
            for i in range(len(nHWr)): # len(nHWr) == len(nCWr)
                count = 0 # for checking ratio in combination(nCWr)
                k = 0
                while k < layer:
                    if(name[k] == 'SQUEEZE\n'): # fallback subgraph
                        f.write('{0}\n'.format(k))
                        f.write('{0}\n'.format(k+1))
                        f.write('{0}\n'.format(self.resource))
                        f.write('{0}\n'.format(nCWr[i][0]))
                        k += 1
                    elif k == 30:
                        f.write('{0}\n'.format(k))
                        f.write('{0}\n'.format(k+1))
                        f.write('{0}\n'.format(0))
                        f.write('{0}\n'.format(0))
                        f.write('{0}\n'.format(-1))
                        f.write('{0}\n'.format(-2))
                        k+=1
                    else:
                        f.write('{0}\n'.format(k))
                        while True:
                            if(name[k] == 'SQUEEZE\n'): # condition has to change by model structure)
                                break
                            else:
                                k += 1
                        f.write('{0}\n'.format(k))
                        f.write('{0}\n'.format(self.resource))
                        f.write('{0}\n'.format(nHWr[i][0]))
            f.close()
            r.close()
    def efficient_combination(self):
            file_name = "../model/efficient/efficient_combination_" + self.name
            f = open(file_name, 'w')
            # layer num
            layer = 118

            # change to layer by test case
            plan_resource = [TF_P_PLAN_CPU, self.resource] #resource type
            
            plan_idx = 0 # subgraph num

            idx = 0
            # change to layer by models
            num = [0] * layer
            name = [0] * layer
            with open('efficient_layer', mode = 'r+', encoding='UTF-8') as r:    
                while True:
                    lines = r.readline()
                    if not lines:
                        break
                    num[idx], name[idx] = lines.split(' ')
                    if(name[idx] == 'RESHAPE\n'): # condition has to change by model structure
                        plan_idx += 1
                    idx += 1
            not_fallback = plan_idx + 1
            # per subgraph's usable resource set
            # repeat = fallback num + 1(subgraph in no fallback layer)
            nREr = list(product(plan_resource, repeat=not_fallback)) # resource product
            nCWr = list(product(plan_ratio_cw, repeat=not_fallback)) # cw product
            nHWr = list(product(plan_ratio_hw, repeat=1)) # hw product
            nRAr = [nCWr, nHWr]
            # need to change model file
            for i in range(len(nHWr)): # len(nHWr) == len(nCWr)
                count = 0 # for checking ratio in combination(nCWr)
                k = 0
                while k < layer: # CW
                    if(k >= 114): # fallback subgraph
                        f.write('{0}\n'.format(k))
                        f.write('{0}\n'.format(k+4))
                        f.write('{0}\n'.format(1))
                        f.write('{0}\n'.format(0))
                        f.write('{0}\n'.format(-1))
                        f.write('{0}\n'.format(-2))
                        break
                    else:
                        f.write('{0}\n'.format(k))
                        while True:
                            if(name[k] == 'AVERAGE_POOL_2D\n'): # condition has to change by model structure)
                                break
                            else:
                                k += 1
                        f.write('{0}\n'.format(k))
                        f.write('{0}\n'.format(self.resource))
                        f.write('{0}\n'.format(nHWr[i][0]))
            f.close()
            r.close()
def main():
    for i in range(len(file_)):
        if(i==0 or i==1 or i==3): #just make file for delegation
            delegate = delegation_combination(i, file_[i])
            delegate.yolo_combination()
            delegate.mobilenet_combination()
            delegate.efficient_combination()

    for i in range(len(file_)):
        if(i==2 or i==4): #just make file for delegation
            co_e = co_execution_combination(i, file_[i])
            co_e.yolo_combination()
            co_e.mobilenet_combination()
            co_e.efficient_combination()
main()