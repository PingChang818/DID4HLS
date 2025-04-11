import os
import copy
import yaml
import random
import pickle
import syn_utils

class GRASP():
    def __init__(self, model, pragmas, data, timeSpentTraining):
        super().__init__()
        self.pragmas = pragmas
        self.data = data
        self.alpha = 0.7
        self.model = model
        self.RCLSynthesisInterval = timeSpentTraining
        self.run()
       
    def run(self):
        while True:
            feature = self.constructGreedyRandomizedSolution()
            feature = self.localSearch(feature)
            ct = 0
            for _ in os.listdir("sim_data"):
                ct += 1
                
            if ct >= 180:
                break

    def sample(self):    
        theta, info = syn_utils.theta_init(self.model.space)
        prob_np = syn_utils.prob(theta, info).cpu().numpy()
        
        return syn_utils.sample(prob_np, info, self.model.space, 1)[0]
    
    def constructGreedyRandomizedSolution(self):
        feature = self.model.p2f([self.sample()])[0]
        for i, values in enumerate(self.model.pvalue):
            RCL = []
            candidates = []
            predictions = []
            bestResourcesXLatency = float('inf')
            for v in values:
                feature[i] = v
                candidates.append(v)
                predictions += self.model.predict([feature])
                resourcesXlatency = predictions[-1][0] * predictions[-1][1]
                if resourcesXlatency < bestResourcesXLatency:
                    bestResourcesXLatency = resourcesXlatency
                    
            for j, prediction in enumerate(predictions):
                resourcesXlatency = prediction[0] * prediction[1]
                if resourcesXlatency < (1 + self.alpha) * bestResourcesXLatency:
                    RCL.append(candidates[j])
                    
            if len(RCL) != 0:
                feature[i] = random.choice(RCL)
            if ((i + 1) % self.RCLSynthesisInterval == 0):
                new_id = syn_utils.sim(self.model.f2p([feature]), self.model.space)
                path = "sim_data/" + str(new_id[0])
                self.data.append(pickle.load(open(path + "/data.pkl", "rb")))
                self.pragmas.append(yaml.load(open(path + "/pragma.yaml", "r"), Loader=yaml.Loader))
                self.model.trainModel(self.pragmas, self.data)
                
        return feature

    def localSearch(self, feature):
        top_metric = float("inf")
        for i, values in enumerate(self.model.pvalue):
            for v in values:
                if feature[i] != v:
                    feature[i] = v
                    prediction = self.model.predict([feature])
                    if top_metric > prediction:
                        top_metric = prediction
                        top = copy.deepcopy(feature)
                    
        new_id = syn_utils.sim([self.model.f2p(top)], self.model.space)
        path = "sim_data/" + str(new_id[0])
        self.data.append(pickle.load(open(path + "/data.pkl", "rb")))
        self.pragmas.append(yaml.load(open(path + "/pragma.yaml", "r"), Loader=yaml.Loader))
        self.model.trainModel(self.pragmas, self.data)
        
        return top