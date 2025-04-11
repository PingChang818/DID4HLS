import os
import yaml
import utils
import torch
import shutil
import random
import pickle
import subprocess
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from model import GAT

def extract(design_list, benchmark, share, tar, is_train):
    os.mkdir(share + "/ll_gnn_dse_" + tar)
    os.mkdir(share + "/code_gnn_dse_" + tar)
    os.mkdir(share + "/graph_gnn_dse_" + tar)
    shutil.copy("data/benchmark/" + benchmark + ".h", share + "/code_gnn_dse_" + tar)
    orders = []
    order = 0
    for design in design_list:
        utils.code4clang(design, benchmark, str(order), share, tar)
        cmd = "clang -O3 -S -emit-llvm "
        cmd = cmd + share + "/code_gnn_dse_" + tar + "/" + str(order) + ".cpp -o " + share + "/ll_gnn_dse_" + tar + "/" + str(order) + ".ll"
        subprocess.run(cmd, shell=True)
        orders.append(order)
        order += 1
        
    utils.cdfg(tar)
    graph = []
    for order in orders:
        pragma = design_list[order]
        cdfg = pickle.load(open(share + "/graph_gnn_dse_" + tar + "/" + str(order) + ".pkl", "rb"))
        for i in cdfg._node.keys():
            if cdfg._node[i]["text"] == "[external]":
                if "interface" in pragma.keys():
                    last = len(cdfg._node)
                    for var in pragma["interface"].keys():
                        cdfg._node[last] = {"text": "port_" + pragma["interface"][var][0]}
                        cdfg._succ[last] = {i: {0: {"flow": 10 + pragma["interface"][var][1]}}}
                        last += 1
                        
                break
            
        graph.append(utils.graph(cdfg, is_train))
        
    shutil.rmtree(share + "/ll_gnn_dse_" + tar)
    shutil.rmtree(share + "/code_gnn_dse_" + tar)
    shutil.rmtree(share + "/graph_gnn_dse_" + tar)
    
    return graph

def init(benchmark, space, share, tar):
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
        
    graph = extract(pragmas, benchmark, share, tar, True)
    
    return graph, data
    
def train(graph, result, benchmark, device, batch_size, eps):
    y_l = []
    y_a = []
    graph_train = []
    result_train = []
    for i, r in enumerate(result):
        if r[0] != -1:
            graph_train.append(graph[i])
            result_train.append(r)
            y_l.append(r[0])
            y_a.append(r[1])
        
    ml = np.mean(y_l)
    sl = np.std(y_l)
    ma = np.mean(y_a)
    sa = np.std(y_a)
    train = []
    for i in range(len(graph_train)):
        dn, det, ded = graph_train[i]
        x = torch.tensor(dn, dtype=torch.long)
        edge_index = torch.tensor(ded, dtype=torch.long)
        edge_attr = torch.tensor(det, dtype=torch.long)
        y = torch.tensor([(result_train[i][0] - ml) / sl, (result_train[i][1] - ma) / sa], dtype=torch.float)
        train.append(Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, y=y).to(device))
        
    criterion = nn.MSELoss()
    gat_l = GAT().to(device)
    gat_a = GAT().to(device)
    optimizer_l = torch.optim.Adam(gat_l.parameters(), lr=1.0e-3)
    optimizer_a = torch.optim.Adam(gat_a.parameters(), lr=1.0e-3)
    loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    lr_scheduler_l = torch.optim.lr_scheduler.MultiStepLR(optimizer_l, milestones=[0.8 * eps], gamma=0.5)
    lr_scheduler_a = torch.optim.lr_scheduler.MultiStepLR(optimizer_a, milestones=[0.8 * eps], gamma=0.5)
    gat_l.train()
    gat_a.train()
    tqdm_train = tqdm(range(eps))
    for epoch in tqdm_train:
        avg_loss_l = 0
        avg_loss_a = 0
        for d in loader:
            loss_l = criterion(gat_l(d), d.y.reshape(-1, 2)[:, 0])
            loss_a = criterion(gat_a(d), d.y.reshape(-1, 2)[:, 1])
            optimizer_l.zero_grad()
            optimizer_a.zero_grad()
            loss_l.backward()
            loss_a.backward()
            avg_loss_l += loss_l.item()
            avg_loss_a += loss_a.item()
            optimizer_l.step()
            optimizer_a.step()
            
        tqdm_train.set_postfix({"avg_loss_l": avg_loss_l / len(train), "avg_loss_a": avg_loss_a / len(train)})
        lr_scheduler_l.step()
        lr_scheduler_a.step()
        
    torch.save(gat_l.state_dict(), "save/gat_l_" + benchmark + ".pth")
    torch.save(gat_a.state_dict(), "save/gat_a_" + benchmark + ".pth")

def sample2syn(benchmark, space, batch_size, device, share, tar):
    print("sample2syn")
    gat_l = GAT().to(device)
    gat_a = GAT().to(device)
    gat_l.load_state_dict(torch.load("save/gat_l_" + benchmark + ".pth"))
    gat_a.load_state_dict(torch.load("save/gat_a_" + benchmark + ".pth"))
    theta, info = utils.theta_init(space)
    prob = utils.prob(theta, info).cpu().numpy()
    samples_pragma = utils.sample(prob, info, space, 15000)
    samples_unique = []
    while len(samples_unique) < 10000:
        pragma = samples_pragma.pop(0)
        if pragma not in samples_unique:
            samples_unique.append(pragma)
        
    pickle.dump(space, open("save/space.pkl", 'wb'))
    pickle.dump(samples_unique, open("save/to_extract.pkl", 'wb'))
    print("extract")
    os.system('cmd /c "python extract.py"')
    orders = []
    for d in os.listdir("graph"):
        orders.append(int(d.split(".")[0]))
        
    orders.sort()
    graph = []
    for order in orders:
        graph += pickle.load(open("graph/" + str(order) + ".pkl", "rb"))
        os.remove("graph/" + str(order) + ".pkl")
        
    pred = []
    for g in graph:
        dn, det, ded = g
        x = torch.tensor(dn, dtype=torch.long)
        edge_index = torch.tensor(ded, dtype=torch.long)
        edge_attr = torch.tensor(det, dtype=torch.long)
        pred.append(Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr).to(device))
        
    loader = DataLoader(pred, batch_size=batch_size, shuffle=False)
    y_l = []
    y_a = []
    ct = 0
    gat_l.eval()
    gat_a.eval()
    print("predict")
    with torch.no_grad():
        for d in loader:
            if len(d) == 1:
                y_l.append(gat_l(d))
                y_a.append(gat_a(d))
            else:
                y_l += gat_l(d)
                y_a += gat_a(d)
            ct += len(d)
            print("{} points predicted".format(ct))
        
    y_l = torch.stack(y_l)
    y_a = torch.stack(y_a)
    p_i = []
    p_a = float("inf")
    for i in torch.sort(y_l)[1].cpu().numpy():
        if p_a > y_a[i]:
            p_a = y_a[i]
            p_i.append(i)
            
    random.shuffle(p_i)
    to_syn = []
    for i in p_i[: 30]:
        to_syn.append(samples_unique[i])
        
    print("sim_final")
    utils.sim(to_syn, space)