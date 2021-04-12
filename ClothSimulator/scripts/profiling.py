import numpy as np
# path = "../noomp.log"
# path = "../omp.log"



def prof(path):
    with open(path) as f:
        data_array = {}
        for i in f.readlines():
            if i.find("cost") != -1:
                key = i.split()[1]
                if False == (key in data_array):
                    data_array[key] = []
                data = data_array[key]
                data.append(float(i.split()[-2]))
            # print(i)
    for key in data_array:
        print(f"{path} :{key} mean = {np.mean(data_array[key])}")
    print('------------')
prof("../omp.log.1")
prof("../omp.log.3")
prof("../omp.log.6")
prof("../omp.log.8")
prof("../omp.log.12")
prof("../noomp.log")