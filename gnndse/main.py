import os
import exe
import yaml
import shutil
import datetime

start = datetime.datetime.now()
device = "cuda:0"
batch_size = 16
eps = 200
tar = "train"
# !!! We run the current project on Windows, but programl and networkx are only supported on Linux. So we call them in Linux. vmware/cdfg.py and vmware/graph.py are placed in the shared folder.
# !!! Specify your share folder directory
share = "... .../share"
if os.path.isfile("emb.pkl"):
    os.remove("emb.pkl")
if os.path.isdir(share + "/ll_gnn_dse_" + tar):
    shutil.rmtree(share + "/ll_gnn_dse_" + tar)
    shutil.rmtree(share + "/code_gnn_dse_" + tar)
    shutil.rmtree(share + "/graph_gnn_dse_" + tar)
if os.path.isdir("sim"):
    shutil.rmtree("sim")
for f in os.listdir("sim_data"):
    shutil.rmtree("sim_data/" + f)
    
for f in os.listdir("save"):
    os.remove("save/" + f)
    
benchmarks = ["correlation", "covariance", "gramSchmidt", "aes", "sort", "stencil"]
benchmark = benchmarks[0]
space = yaml.safe_load(open("data/space/" + benchmark + ".yaml", "r"))
graph, data = exe.init(benchmark, space, share, tar)
exe.train(graph, data, benchmark, device, batch_size, eps)
exe.sample2syn(benchmark, space, batch_size, device, share, tar)
end = datetime.datetime.now()
print("time:")
print((end-start).seconds)
