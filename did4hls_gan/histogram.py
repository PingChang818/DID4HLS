import os
import json
import numpy as np
import matplotlib.pyplot as plt

hls_path = "C:/Users/Lab/Vitis_projects/spaces/"
space_path = "C:/Users/Lab/Vivado_projects/spaces/"

power = np.zeros(1024)
interval = np.zeros(1024)
for space_id in os.listdir("data/spaces"):
    for point in os.listdir("data/spaces/" + space_id):
        if point.split(".")[1] != "cpp":
            continue
        i = point.split(".")[0]
        path = hls_path + space_id + "/10/" + i + "/solution/solution_data.json"
        if os.path.isfile(path):
            with open(path, "r") as f:
                interval[int(i)] = int(json.load(f)["ModuleInfo"]["Metrics"]["correlation"]["Latency"]["PipelineIIMax"])
        else:
            print("vitis file not found")
            break
        
        vivado_path = space_path + space_id + "/10/" + i + "/" + i + ".runs/impl_1/correlation_power_routed.rpt"
        if os.path.isfile(vivado_path):
            with open(vivado_path, "r") as f:
                rpt = f.readlines()
                for l in rpt:
                    if "Dynamic (W)" in l:
                        power[int(i)] = float(l.split("|")[2].strip())
        # else:
        #     print("vivado file not found")
        #     break
        
# c = 0
# for i in range(len(interval)):
#     if interval[i] == 0:
#         print(i)
#         c += 1
# print(c)

h = []
for i in range(len(power)):
    if power[i] != 0:
        h.append(power[i] * interval[i])
    else:
        h.append(0)

plt.hist(h, bins=40)
plt.show()