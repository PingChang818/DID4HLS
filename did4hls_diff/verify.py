import os
import torch
import pickle
import numpy as np
from model import GAT, CVAE
from torch_geometric.loader import DataLoader

device = "cuda:0"
batch_size = 25
theta = pickle.load(open("save/0/theta.pkl", "rb"))
gatl = GAT().to(device)
gata = GAT().to(device)
cvael = CVAE(len(theta)).to(device)
cvaea = CVAE(len(theta)).to(device)
it = 12
gatl.load_state_dict(torch.load("save/"+ str(it) +"/gatl.pth"))
gata.load_state_dict(torch.load("save/"+ str(it) +"/gata.pth"))
y_l = pickle.load(open("save/"+ str(it) +"/y_l.pkl", "rb"))
y_a = pickle.load(open("save/"+ str(it) +"/y_a.pkl", "rb"))
y_lt = torch.stack(y_l)
y_at = torch.stack(y_a)
ml = torch.mean(y_lt)
sl = torch.sqrt(torch.var(y_lt))
ma = torch.mean(y_at)
sa = torch.sqrt(torch.var(y_at))

s_it = []
for it in os.listdir("it"):
    s_it += pickle.load(open("it/" + it, "rb"))
    
s_it = np.unique(s_it).tolist()
    
# s = []
# for p in os.listdir("sim_data"):
#     s.append(p)
    
# idx = np.arange(len(s))
# np.random.shuffle(idx)

# ct = 0
# test = []
# for i in idx:
#     if not s[i] in s_it:
#         d = pickle.load(open("sim_data/" + s[i] + "/data.pkl", "rb"))
#         d.y[0] = (d.y[0] - ml) / sl
#         d.y[1] = (d.y[1] - ma) / sa
#         test.append(d.to(device))
#         ct += 1
#     if ct >= 500:
#         break
    
test = []   
for p in s_it:
    d = pickle.load(open("sim_data/" + p + "/data.pkl", "rb"))
    d.y[0] = (d.y[0] - ml) / sl
    d.y[1] = (d.y[1] - ma) / sa
    test.append(d.to(device))
        
        
pred_l = []
pred_a = []
gt_l = []
gt_a = []
loader = DataLoader(test, batch_size=batch_size, shuffle=True)
print("testing...")
gatl.eval()
gata.eval()
with torch.no_grad():
    for d in loader:
        pred_l.append(gatl.predict(gatl(d)).squeeze())
        pred_a.append(gata.predict(gata(d)).squeeze())
        gt_l.append(d.y.reshape(-1, 2)[:, 0])
        gt_a.append(d.y.reshape(-1, 2)[:, 1])
        z = torch.randn(25, 64, device=device)

pickle.dump(pred_l, open("test/verify/pred_l.pkl", 'wb'))
pickle.dump(pred_a, open("test/verify/pred_a.pkl", 'wb'))
pickle.dump(gt_l, open("test/verify/gt_l.pkl", 'wb'))
pickle.dump(gt_a, open("test/verify/gt_a.pkl", 'wb'))
