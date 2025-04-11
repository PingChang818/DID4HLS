import os
import yaml
import shutil
import pickle
import syn_utils
from heuristics.impl.GRASP import GRASP
from predictor.estimators.randomforest.randomForest import RandomForestEstimator
import datetime

start = datetime.datetime.now()
if os.path.isdir("sim"):
    shutil.rmtree("sim")
for f in os.listdir("sim_data"):
    shutil.rmtree("sim_data/" + f)
    
benchmarks = ["correlation", "covariance", "gramSchmidt", "aes", "sort", "stencil"]
# !!! Specify which benchmark to optimize
space = yaml.safe_load(open("data/space/" + benchmarks[0] + ".yaml", "r"))
timer_train = 600
timer_search = 60
pickle.dump(timer_train, open("hls_timer.pkl", 'wb'))
theta, info = syn_utils.theta_init(space)
prob_np = syn_utils.prob(theta, info).cpu().numpy()
unique_pragma = []
train_size = 120
while len(unique_pragma) < train_size:
    samples_pragma = syn_utils.sample(prob_np, info, space, train_size + 10)
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
    syn_utils.sim(to_syn, space)
    
data = []
pragmas = []
for d in os.listdir("sim_data"):
    data.append(pickle.load(open("sim_data/" + d + "/data.pkl", "rb")))
    pragmas.append(yaml.load(open("sim_data/" + d + "/pragma.yaml", "r"), Loader=yaml.Loader))
    
model = RandomForestEstimator(space)
model.trainModel(pragmas, data)
pickle.dump(timer_search, open("hls_timer.pkl", 'wb'))
GRASP(model, pragmas, data, 8, start)