import os
import pickle
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

# !!! Specify the results of which benchmark to plot
bid = 0
tick_x = [[100000, 200000, 300000], [100000, 200000, 300000], [200000, 400000, 600000], [1000, 2000], [100000, 200000, 300000], [30000, 60000, 90000]]
tick_y = [[0.02, 0.04, 0.06], [0.02, 0.04, 0.06], [0.01, 0.02], [0.003, 0.005, 0.007], [0.01, 0.02], [0.01, 0.02, 0.03]]
benchmarks = ["correlation", "covariance", "gramSchmidt", "aes", "sort", "stencil"]
benchmark_name = [r"$correlation$", r"$covariance$", r"$gramSchmidt$", r"$aes$", r"$sort\_radix$", r"$stencil\_3d$"]
benchmark = benchmarks[bid]
model_name = {'did4hls': r'$DID4HLS$', 'autohls': r'$AutoHLS$', 'grasp': r'$GRASP$', 'saml': r'$SA-ML$', 'gnndse': r'$GNN-DSE$'}
ablation = ["gan", "diff", "llvm"]
font = {'size': 14}
plt.rc('font', **font)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.linewidth'] = 1
l = []
a = []
for m in os.listdir("result/" + benchmark):
    for i in os.listdir("result/" + benchmark + "/" + m):
        d = pickle.load(open("result/" + benchmark + "/" + m + "/" + i + "/data.pkl", "rb"))
        if isinstance(d, list):
            la = d
        else:
            la = d.y.cpu().numpy().tolist()
        l.append(la[0])
        a.append(la[1])
        
p_i = []
p_a = float("inf")
for i in sorted(range(len(l)), key=lambda k: l[k]):
    if p_a > a[i]:
        p_a = a[i]
        p_i.append(i)
        
p_l = []
p_a = []
for i in p_i:
    p_l.append(l[i])
    p_a.append(a[i])
    
pickle.dump([p_l, p_a], open("pareto/" + benchmark + ".pkl", "wb"))
p_l_ref, p_a_ref = pickle.load(open("pareto/" + benchmark + ".pkl", "rb"))
data_m = []
model = []
p_i_m = []
for m in os.listdir("result/" + benchmark):
    # !!! Ablation results excluded
    if m in ablation:
        continue
    
    model.append(m)
    l = []
    a = []
    for i in os.listdir("result/" + benchmark + "/" + m):
        d = pickle.load(open("result/" + benchmark + "/" + m + "/" + i + "/data.pkl", "rb"))
        if isinstance(d, list):
            la = d
        else:
            la = d.y.cpu().numpy().tolist()
        l.append(la[0])
        a.append(la[1])
        
    data_m.append([l, a])
    
adrs = [0] * len(data_m)
for i_m in range(len(data_m)):
    l, a = data_m[i_m]
    p_i = []
    p_a = float("inf")
    for i in sorted(range(len(l)), key=lambda k: l[k]):
        if p_a > a[i]:
            p_a = a[i]
            p_i.append(i)
            
    p_i_m.append(p_i)
    n_p = len(p_l_ref)
    for i_gama in range(n_p):
        comps = []
        for i_omega in p_i:
            dist = abs((p_l_ref[i_gama] - l[i_omega]) / p_l_ref[i_gama])
            dist_a = abs((p_a_ref[i_gama] - a[i_omega]) / p_a_ref[i_gama])
            if dist < dist_a:
                dist = dist_a
            comps.append(dist)
            
        adrs[i_m] += min(comps)
        
    adrs[i_m] /= n_p
    
print("")
for i, m in enumerate(model):
    print("{}: {}".format(m, round(adrs[i], 3)))
    
color = ['sienna', 'teal']
marker = ['x', '+']
plt.scatter(p_l_ref, p_a_ref, s=80, linewidths=2, facecolors='none', edgecolors='salmon')
ct = 0
m = []
for i_m in sorted(range(len(adrs)), key=lambda k: adrs[k])[: 2]:
    m.append(model[i_m])
    l, a = data_m[i_m]
    p_l = []
    p_a = []
    for i in p_i_m[i_m]:
        p_l.append(l[i])
        p_a.append(a[i])
        
    plt.scatter(p_l, p_a, s=50, linewidths=2, c=color[ct], marker=marker[ct])
    ct += 1
    
formatter_x = ticker.ScalarFormatter(useMathText=True)
formatter_x.set_powerlimits((1, -1))
formatter_y = ticker.ScalarFormatter(useMathText=True)
formatter_y.set_powerlimits((1, -1))
ax = plt.gca()
ax.xaxis.set_major_formatter(formatter_x)
ax.yaxis.set_major_formatter(formatter_y)
plt.title(benchmark_name[bid])
plt.xlabel('latency')
plt.ylabel('SNRU')
plt.legend(['reference', model_name[m[0]], model_name[m[1]]])
plt.xticks(tick_x[bid])
plt.yticks(tick_y[bid])
plt.show()