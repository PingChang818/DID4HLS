import os
import yaml
import utils
import shutil
import pickle
import datetime

start = datetime.datetime.now()
utils.reboot()
# !!! We run the current project on Windows, but programl and networkx are only supported on Linux. So we call them in Linux. vmware/cdfg.py and vmware/graph.py are placed in the shared folder.
# !!! Specify your share folder directory
for f in os.listdir("... .../share"):
    if f != "cdfg.py" and f != "graph.py":
        # !!! Specify your share folder directory
        shutil.rmtree("... .../share/" + f)
        
for f in os.listdir("save"):
    os.remove("save/" + f)
    
for f in os.listdir("sa"):
    os.remove("sa/" + f)
    
for f in os.listdir("sa_log"):
    os.remove("sa_log/" + f)
    
for f in os.listdir("sim_data"):
    shutil.rmtree("sim_data/" + f)
    
benchmarks = ["correlation", "covariance", "gramSchmidt", "aes", "sort", "stencil"]
# !!! Specify which benchmark to optimize
space = yaml.safe_load(open("data/space/" + benchmarks[0] + ".yaml", "r"))
pickle.dump(space, open("save/space.pkl", 'wb'))
theta, info = utils.theta_init(space)
prob_np = utils.prob(theta, info).cpu().numpy()
unique_pragma = []
train_size = 150
while len(unique_pragma) < train_size:
    samples_pragma = utils.sample(prob_np, info, space, train_size + 10)
    unique_pragma = []
    while len(samples_pragma) > 0:
        pragma = samples_pragma.pop(0)
        if pragma not in samples_pragma:
            unique_pragma.append(pragma)
            
unique_pragma = unique_pragma[: train_size]
while len(unique_pragma) > 0:
    if len(unique_pragma) > 30:
        to_syn = unique_pragma[: 30]
        unique_pragma = unique_pragma[30 :]
    else:
        to_syn = unique_pragma
        unique_pragma = []
    utils.sim(to_syn, space)
    
data = []
pragmas = []
for d in os.listdir("sim_data"):
    data.append(pickle.load(open("sim_data/" + d + "/data.pkl", "rb")))
    pragmas.append(yaml.load(open("sim_data/" + d + "/pragma.yaml", "r"), Loader=yaml.Loader))
    
# !!! Specify your share folder directory
features = utils.extract(pragmas, space, "... .../share", "extract")
utils.train(features, data)
print("annealing...")
os.system('cmd /c "python opt.py"')
to_syn = []
for p in os.listdir("sa"):
    to_syn.append(yaml.load(open("sa/" + p, "r"), Loader=yaml.Loader))
    
utils.sim(to_syn, space)
end = datetime.datetime.now()
print("time:")
print((end-start).seconds)