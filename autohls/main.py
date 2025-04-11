import os
import yaml
import torch
import random
import shutil
import pickle
import optuna
import syn_utils
from estimator import Estimator
from optuna.samplers import NSGAIISampler
import datetime

def objective(trial, est):
    feature = []
    for i, values in enumerate(est.pvalue):
        feature.append(trial.suggest_categorical(est.ptype[i], values))
        
    predictions = est.get_prediction(torch.tensor([feature] * 10, dtype=torch.float32).cuda())
    
    return torch.mean(predictions, dim = 0).cpu().numpy().tolist()

def get_unique(samples):
    simulated = []
    for d in os.listdir("sim_data"):
        simulated.append(yaml.load(open("sim_data/" + d + "/pragma.yaml", "r"), Loader=yaml.Loader))
        
    unique = []
    it_size = 30
    while len(unique) < it_size:
        unique = []
        while len(samples) > 0:
            pragma = samples.pop(0)
            ns = True
            for s in simulated:
                if pragma == s:
                    ns = False
                    break
                
            if ns and pragma not in samples_pragma:
                unique.append(pragma)
                
    unique = unique[: it_size]
    
    return unique

start = datetime.datetime.now()
if os.path.isdir("sim"):
    shutil.rmtree("sim")
for f in os.listdir("sim_data"):
    shutil.rmtree("sim_data/" + f)
    
benchmarks = ["correlation", "covariance", "gramSchmidt", "aes", "sort", "stencil"]
# !!! Specify which benchmark to optimize
space = yaml.safe_load(open("data/space/" + benchmarks[0] + ".yaml", "r"))
theta, info = syn_utils.theta_init(space)
prob_np = syn_utils.prob(theta, info).cpu().numpy()
samples_pragma = []
print("iteration: 1")
unique_pragma = get_unique(syn_utils.sample(prob_np, info, space, 40))
samples_pragma += unique_pragma
syn_utils.sim(unique_pragma, space)
data = []
pragmas = []
for d in os.listdir("sim_data"):
    data.append(pickle.load(open("sim_data/" + d + "/data.pkl", "rb")))
    pragmas.append(yaml.load(open("sim_data/" + d + "/pragma.yaml", "r"), Loader=yaml.Loader))
    
for it in range(5):
    print("iteration: {}".format(it + 2))
    invalid_pragma = []
    for p in samples_pragma:
        if p not in pragmas:
            invalid_pragma.append(p)
            
    est = Estimator(space)
    est.trainModel(pragmas, data, invalid_pragma)
    study = optuna.create_study(directions=['minimize', 'minimize'], sampler=NSGAIISampler())
    study.optimize(lambda trial: objective(trial, est), n_trials=100)
    pareto_front = study.best_trials
    pareto_features = []
    for point in pareto_front:
        feature = [-1] * len(est.ptype)
        for key in enumerate(point.params):
            feature[est.ptype.index(key[1])] = point.params[key[1]]
            
        pareto_features.append(feature)
        
    random.shuffle(pareto_features)
    to_syn = []
    for point_feature in pareto_features:
        point_pragma = est.f2p([point_feature])[0]
        if not point_pragma in samples_pragma and est.get_score([point_feature]) > 0.5:
            samples_pragma.append(point_pragma)
            to_syn.append(point_pragma)
        if len(to_syn) >= 30:
            break
    
    new_id = syn_utils.sim(to_syn, space)
    for i in new_id:
        data.append(pickle.load(open("sim_data/" + str(i) + "/data.pkl", "rb")))
        pragmas.append(yaml.load(open("sim_data/" + str(i) + "/pragma.yaml", "r"), Loader=yaml.Loader))
        
end = datetime.datetime.now()
print("time:")
print((end-start).seconds)