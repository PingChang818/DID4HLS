import copy
import torch
from tqdm import tqdm
from models import Scorer, Predictor

class Estimator:
    def __init__(self, space):
        ptype = []
        pvalue = []
        pragma_start = {}
        for t in enumerate(space):
            if t[1] == "function":
                pragma_start["inline"] = {}
                for fc in space[t[1]]:
                    ptype.append("inline&" + fc)
                    pvalue.append([True, False])
                    pragma_start["inline"][fc] = False
                    
            elif t[1] == "interface":
                pragma_start["interface"] = {}
                for i, itf in enumerate(space[t[1]]):
                    ptype.append("interface&" + itf)
                    pvalue.append(space["p_itf"][i])
                    ptype.append("interface&" + itf + "&type")
                    pvalue.append([1, 0])
                    pragma_start["interface"][itf] = ['block', 1]
                    
            elif t[1] == "nest":
                pragma_start["ii"] = [-1] * len(space["nest"])
                pragma_start["unroll"] = [1] * len(space["nest"])
                for i in range(len(space[t[1]])):
                    ptype.append("ii&" + str(i))
                    n = len(space["ii"][i]) * space["pipeline"][i] / (1 - space["pipeline"][i])
                    n = int(round(n))
                    if n == 0:
                        n = 1
                    pvalue.append(([-1] * n) + space["ii"][i])
                    ptype.append("unroll&" + str(i))
                    pvalue.append(space["unroll"][i])
                    
        self.space = space
        self.ptype = ptype
        self.pvalue = pvalue
        self.pragma_start = pragma_start
        self.predictor = Predictor(len(ptype)).cuda()
        self.scorer = Scorer(len(ptype)).cuda()
        
    def p2f(self, pragmas):
        features = []
        for pragma in pragmas:
            feature = [-1] * len(self.ptype)
            for p in enumerate(pragma):
                if p[1] == "inline":
                    for fc in enumerate(pragma[p[1]]):
                        if pragma[p[1]][fc[1]]:
                            feature[self.ptype.index("inline&" + fc[1])] = 1
                        else:
                            feature[self.ptype.index("inline&" + fc[1])] = 0
                            
                elif p[1] == "interface":
                    for itf in enumerate(pragma[p[1]]):
                        feature[self.ptype.index("interface&" + itf[1])] = pragma[p[1]][itf[1]][1]
                        if pragma[p[1]][itf[1]][0] == "block":
                            feature[self.ptype.index("interface&" + itf[1] + "&type")] = 1
                        else:
                            feature[self.ptype.index("interface&" + itf[1] + "&type")] = 0
                        
                elif p[1] == "ii":
                    for i, ii in enumerate(pragma[p[1]]):
                        feature[self.ptype.index("ii&" + str(i))] = ii
                        
                elif p[1] == "unroll":
                    for i, unroll in enumerate(pragma[p[1]]):
                        feature[self.ptype.index("unroll&" + str(i))] = unroll
                        
            features.append(feature)
            
        return features
    
    def f2p(self, features):
        pragmas = []
        for feature in features:
            pragma = copy.deepcopy(self.pragma_start)
            for i, f in enumerate(feature):
                tl = self.ptype[i].split("&")
                if tl[0] == "inline":
                    if f == 1:
                        pragma[tl[0]][tl[1]] = True
                    else:
                        pragma[tl[0]][tl[1]] = False
                elif tl[0] == "interface":
                    if len(tl) > 2:
                        if f == 1:
                            pragma[tl[0]][tl[1]][0] = "block"
                        else:
                            pragma[tl[0]][tl[1]][0] = "cyclic"
                    else:
                        pragma[tl[0]][tl[1]][1] = f
                elif tl[0] == "ii":
                    pragma[tl[0]][int(tl[1])] = f
                elif tl[0] == "unroll":
                    pragma[tl[0]][int(tl[1])] = f
                
            pragmas.append(pragma)
            
        return pragmas
        
    def trainModel(self, pragmas, data, invalid_pragma):
        self.predictor.train()
        self.scorer.train()
        criterion = torch.nn.MSELoss()
        features = torch.tensor(self.p2f(pragmas), dtype = torch.float32).cuda()
        invalid_features = torch.tensor(self.p2f(invalid_pragma), dtype = torch.float32).cuda()
        data = torch.tensor(data, dtype = torch.float32).cuda()
        isvalid = torch.ones(len(pragmas) + len(invalid_pragma), dtype = torch.float32).cuda()
        isvalid[len(pragmas) :] = 0
        tqdm_predictor = tqdm(range(300))
        optimizer_predictor = torch.optim.Adam(self.predictor.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_predictor, milestones=[0.8 * 200], gamma=0.5)
        for epoch in tqdm_predictor:
            loss = criterion(self.predictor(features), data)
            optimizer_predictor.zero_grad()
            loss.backward()
            optimizer_predictor.step()
            tqdm_predictor.set_postfix({"loss": loss.item(), "lr": optimizer_predictor.param_groups[0]["lr"]})
            lr_scheduler.step()
            
        tqdm_scorer = tqdm(range(300))
        optimizer_scorer = torch.optim.Adam(self.scorer.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_scorer, milestones=[0.8 * 200], gamma=0.5)
        for epoch in tqdm_scorer:
            loss = criterion(self.scorer(torch.cat((features, invalid_features))), isvalid)
            optimizer_scorer.zero_grad()
            loss.backward()
            optimizer_scorer.step()
            tqdm_predictor.set_postfix({"loss": loss.item(), "lr": optimizer_scorer.param_groups[0]["lr"]})
            lr_scheduler.step()
            
    def get_prediction(self, pragmas):
        x = torch.tensor(self.p2f(pragmas), dtype = torch.float32).cuda()
        self.predictor.eval()
        with torch.no_grad():
            prediction = self.predictor(x)
            
        return prediction
    
    def get_score(self, pragmas):
        x = torch.tensor(self.p2f(pragmas), dtype = torch.float32).cuda()
        self.scorer.eval()
        with torch.no_grad():
            score = self.scorer(x)
            
        return score