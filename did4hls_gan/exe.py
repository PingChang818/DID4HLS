import os
import sys
import yaml
import copy
import torch
import utils
import shutil
import pickle
import torch.nn as nn
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from matplotlib import pyplot as plt
from model import GAT, Generator, Discriminator

def dse(trial, space, config):
    theta_p = []
    prob_it = []
    samples_pragma_p = []
    simulated_it = []
    y_l = []
    y_a = []
    design = []
    design_id = []
    gatl = None
    gata = None
    lambda_p = config["lambda_p"]
    theta, info = utils.theta_init(space)
    len_theta = len(theta)
    criterion = nn.MSELoss()
    for i in range(len(lambda_p)):
        theta_p.append(copy.deepcopy(theta).to(config["device"]))
        
    for it in range(config["n_it"]):
        print("trial: {}   iteration: {}".format(trial + 1, it + 1))
        prob_p = []
        prob_np_p = []
        for i in range(len(lambda_p)):
            p = utils.prob(copy.deepcopy(theta_p[i]), info)
            prob_p.append(p)
            prob_np_p.append(copy.deepcopy(p).cpu().numpy())
            
        prob_it.append(prob_p)
        samples_pragma = []
        samples_pragma_it = []
        for i_p in range(len(lambda_p)):
            samples_pragma_it.append(utils.sample(prob_np_p[i_p], info, space, config["it_size"]))
            samples_pragma += samples_pragma_it[i_p]
            
        samples_pragma_p.append(samples_pragma_it)
        new_id = utils.sim(samples_pragma, space, config)
        simulated_it.append(new_id)
        for i in new_id:
            d = pickle.load(open("sim_data/" + str(i) + "/data.pkl", "rb"))
            design.append(d)
            design_id.append(i)
            y_l.append(d.y[0])
            y_a.append(d.y[1])
            
        train = copy.deepcopy(design)
        train_id = copy.deepcopy(design_id)
        y_l_it = copy.deepcopy(y_l)
        y_a_it = copy.deepcopy(y_a)
        y_lt = torch.stack(y_l_it)
        y_at = torch.stack(y_a_it)
        ml = torch.mean(y_lt)
        sl = torch.sqrt(torch.var(y_lt))
        ma = torch.mean(y_at)
        sa = torch.sqrt(torch.var(y_at))
        id_del = []
        th = config["truncate"]
        for id_d, d in enumerate(train):
            if (d.y[0] - ml) / sl > th or (d.y[1] - ma) / sa > th:
                id_del = [id_d] + id_del
                
        for i in id_del:
            train.pop(i)
            train_id.pop(i)
            y_l_it.pop(i)
            y_a_it.pop(i)
            
        y_lt = torch.stack(y_l_it)
        y_at = torch.stack(y_a_it)
        ml = torch.mean(y_lt)
        sl = torch.sqrt(torch.var(y_lt))
        ma = torch.mean(y_at)
        sa = torch.sqrt(torch.var(y_at))
        for d in train:
            d.y[0] = (d.y[0] - ml) / sl
            d.y[1] = (d.y[1] - ma) / sa
            
        if it + 1 >= config["n_it"] or len(new_id) == 0:
            pickle.dump(simulated_it, open("save/it.pkl", "wb"))
            plot()
            break
        loader = DataLoader(train, batch_size=config["batch_size"], shuffle=True)
        print("{} train data".format(len(train)))
        print("training latency predictor:")
        gatl = GAT().to(config["device"])
        gatl.train()
        optimizer_pl = torch.optim.Adam(gatl.parameters(), lr=config["lr_gat"])
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_pl, milestones=[0.8 * config["eps_gat"]], gamma=0.5)
        tqdm_gatl = tqdm(range(config["eps_gat"]))
        min_loss = float("inf")
        for epoch in tqdm_gatl:
            sum_loss = 0
            t = 0
            for d in loader:
                if len(d) == 1:
                    loss = criterion(gatl.predict(gatl(d))[0][0], d.y[0])
                else:
                    loss = criterion(gatl.predict(gatl(d)).squeeze(), d.y.reshape(-1, 2)[:, 0].squeeze())
                optimizer_pl.zero_grad()
                loss.backward()
                sum_loss += loss.item() * d.num_graphs
                t += d.num_graphs
                optimizer_pl.step()
                
            avg_loss = sum_loss / t
            if min_loss > avg_loss:
                min_loss = avg_loss
                torch.save(gatl.state_dict(), "save/gatl.pth")
            tqdm_gatl.set_postfix({"loss": avg_loss, "lr": optimizer_pl.param_groups[0]["lr"]})
            lr_scheduler.step()
        
        print("min_loss = {:.4e}".format(min_loss))
        gatl.load_state_dict(torch.load("save/gatl.pth"))
        print("training area predictor:")
        gata = GAT().to(config["device"])
        gata.train()
        optimizer_pa = torch.optim.Adam(gata.parameters(), lr=1.0e-3)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_pa, milestones=[0.8 * config["eps_gat"]], gamma=0.5)
        tqdm_gata = tqdm(range(config["eps_gat"]))
        min_loss = float("inf")
        for epoch in tqdm_gata:
            sum_loss = 0
            t = 0
            for d in loader:
                if len(d) == 1:
                    loss = criterion(gata.predict(gata(d))[0][0], d.y[1])
                else:
                    loss = criterion(gata.predict(gata(d)).squeeze(), d.y.reshape(-1, 2)[:, 1].squeeze())
                optimizer_pa.zero_grad()
                loss.backward()
                sum_loss += loss.item() * d.num_graphs
                t += d.num_graphs
                optimizer_pa.step()
                
            avg_loss = sum_loss / t
            if min_loss > avg_loss:
                min_loss = avg_loss
                torch.save(gata.state_dict(), "save/gata.pth")
            tqdm_gata.set_postfix({"loss": avg_loss, "lr": optimizer_pa.param_groups[0]["lr"]})
            lr_scheduler.step()
            
        print("min_loss = {:.4e}".format(min_loss))
        gata.load_state_dict(torch.load("save/gata.pth"))
        gt_l = []
        gt_a = []
        pred_l = []
        pred_a = []
        gatl.eval()
        gata.eval()
        with torch.no_grad():
            for d in loader:
                if len(d) == 1:
                    pred_l += gatl.predict(gatl(d))[0]
                    pred_a += gata.predict(gata(d))[0]
                    gt_l += [d.y[0]]
                    gt_a += [d.y[1]]
                else:
                    pred_l += gatl.predict(gatl(d)).squeeze()
                    pred_a += gata.predict(gata(d)).squeeze()
                    gt_l += d.y.reshape(-1, 2)[:, 0].squeeze()
                    gt_a += d.y.reshape(-1, 2)[:, 1].squeeze()
                    
        pred_l = torch.stack(pred_l).squeeze().cpu().numpy()
        pred_a = torch.stack(pred_a).squeeze().cpu().numpy()
        gt_l = torch.stack(gt_l).squeeze().cpu().numpy()
        gt_a = torch.stack(gt_a).squeeze().cpu().numpy()
        plt.scatter(pred_l, gt_l, s = 0.5)
        plt.legend(["gatl_" + str(it + 1)])
        plt.gcf().savefig("plot/gatl_" + str(it + 1) + ".png", dpi = 300)
        plt.gcf().clf()
        plt.scatter(pred_a, gt_a, s = 0.5)
        plt.legend(["gata_" + str(it + 1)])
        plt.gcf().savefig("plot/gata_" + str(it + 1) + ".png", dpi = 300)
        plt.gcf().clf()
        feat_l_ind = []
        feat_a_ind = []
        prob_ind = []
        with torch.no_grad():
            for d in train:
                prob_ind.append(d.prob)
                for d_i in DataLoader([d]):
                    feat_l_ind.append(gatl(d_i)[0])
                    feat_a_ind.append(gata(d_i)[0])
                    
        feat_l = []
        feat_a = []
        cond = []
        for i_it in range(len(simulated_it)):
            for i in simulated_it[i_it]:
                pragma = yaml.load(open("sim_data/" + str(i) + "/pragma.yaml", "r"), Loader=yaml.Loader)
                for i_p in range(len(lambda_p)):
                    if pragma in samples_pragma_p[i_it][i_p] and i in train_id:
                        i_ind = train_id.index(i)
                        feat_l.append(feat_l_ind[i_ind])
                        feat_a.append(feat_a_ind[i_ind])
                        cond.append(prob_it[i_it][i_p])
                    
        feat_l = torch.stack(feat_l + feat_l_ind)
        feat_a = torch.stack(feat_a + feat_a_ind)
        cond = torch.stack(cond + prob_ind)
        zeros = torch.zeros(len(cond), device=cond.device)
        ones = torch.ones(len(cond), device=cond.device)
        eps = 1000
        print("training latency estimator:")
        genl = Generator(len_theta).to(config["device"])
        disl = Discriminator(len_theta).to(config["device"])
        genl.train()
        disl.train()
        optimizer_genl = torch.optim.Adam(genl.parameters(), lr=1e-3)
        optimizer_disl = torch.optim.Adam(disl.parameters(), lr=1e-3)
        lr_scheduler_genl = torch.optim.lr_scheduler.MultiStepLR(optimizer_genl, milestones=[0.8 * eps], gamma=0.5)
        lr_scheduler_disl = torch.optim.lr_scheduler.MultiStepLR(optimizer_disl, milestones=[0.8 * eps], gamma=0.5)
        tqdm_ganl = tqdm(range(eps))
        min_gen_loss = float("inf")
        for epoch in tqdm_ganl:
            optimizer_genl.zero_grad()
            optimizer_disl.zero_grad()
            feat_gen = genl(cond)
            label_gen = disl(feat_gen, cond).squeeze()
            label_true = disl(feat_l, cond).squeeze()
            dis_loss = (criterion(label_gen, zeros) + criterion(label_true, ones)) / 2
            dis_loss.backward()
            optimizer_disl.step()
            feat_gen = genl(cond)
            label_gen = disl(feat_gen, cond).squeeze()
            gen_loss = criterion(label_gen, ones)
            gen_loss.backward()
            optimizer_genl.step()
            tqdm_ganl.set_postfix({"gen_loss": gen_loss.item(), "dis_loss": dis_loss.item(), "lr_gen": optimizer_genl.param_groups[0]["lr"], "lr_dis": optimizer_disl.param_groups[0]["lr"]})
            lr_scheduler_genl.step()
            lr_scheduler_disl.step()
            if min_gen_loss > gen_loss.item():
                min_gen_loss = gen_loss.item()
                torch.save(genl.state_dict(), "save/genl.pth")
                torch.save(disl.state_dict(), "save/disl.pth")
                
        print("min_gen_loss = {:.4e}".format(min_gen_loss))
        genl.load_state_dict(torch.load("save/genl.pth"))
        disl.load_state_dict(torch.load("save/disl.pth"))
        print("training area estimator:")
        gena = Generator(len_theta).to(config["device"])
        disa = Discriminator(len_theta).to(config["device"])
        gena.train()
        disa.train()
        optimizer_gena = torch.optim.Adam(gena.parameters(), lr=1e-3)
        optimizer_disa = torch.optim.Adam(disa.parameters(), lr=1e-3)
        lr_scheduler_gena = torch.optim.lr_scheduler.MultiStepLR(optimizer_gena, milestones=[0.8 * eps], gamma=0.5)
        lr_scheduler_disa = torch.optim.lr_scheduler.MultiStepLR(optimizer_disa, milestones=[0.8 * eps], gamma=0.5)
        tqdm_gana = tqdm(range(eps))
        min_gen_loss = float("inf")
        for epoch in tqdm_gana:
            optimizer_gena.zero_grad()
            optimizer_disa.zero_grad()
            feat_gen = gena(cond)
            label_gen = disa(feat_gen, cond).squeeze()
            label_true = disa(feat_a, cond).squeeze()
            dis_loss = (criterion(label_gen, zeros) + criterion(label_true, ones)) / 2
            dis_loss.backward()
            optimizer_disa.step()
            feat_gen = gena(cond)
            label_gen = disa(feat_gen, cond).squeeze()
            gen_loss = criterion(label_gen, ones)
            gen_loss.backward()
            optimizer_gena.step()
            tqdm_gana.set_postfix({"gen_loss": gen_loss.item(), "dis_loss": dis_loss.item(), "lr_gen": optimizer_gena.param_groups[0]["lr"], "lr_dis": optimizer_disa.param_groups[0]["lr"]})
            lr_scheduler_gena.step()
            lr_scheduler_disa.step()
            if min_gen_loss > gen_loss.item():
                min_gen_loss = gen_loss.item()
                torch.save(gena.state_dict(), "save/gena.pth")
                torch.save(disa.state_dict(), "save/disa.pth")
                
        print("min_gen_loss = {:.4e}".format(min_gen_loss))
        gena.load_state_dict(torch.load("save/gena.pth"))
        disa.load_state_dict(torch.load("save/disa.pth"))
        genl.eval()
        disl.eval()
        gena.eval()
        disa.eval()
        print("optimizing theta:")
        eps_opt = (it * 2 + 1) * config["eps_opt"]
        for i_p in range(len(lambda_p)):
            theta_p[i_p].requires_grad = True
            optimizer_t = torch.optim.Adam([theta_p[i_p]], lr=config["lr_opt"])
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_t, milestones=[0.8 * eps_opt], gamma=0.5)
            print("lambda: {}".format(lambda_p[i_p]))
            tqdm_opt = tqdm(range(eps_opt))
            for epoch in tqdm_opt:
                prob_nn = utils.prob(torch.clone(theta_p[i_p]), info)
                prob_nn = prob_nn.expand((100, len(prob_nn)))
                yl = genl(prob_nn)
                ya = gena(prob_nn)
                y = lambda_p[i_p] * yl + (1 - lambda_p[i_p]) * ya
                y0 = torch.full_like(y, -1 * (lambda_p[i_p] * ml / sl + (1 - lambda_p[i_p]) * ma / sa))
                loss = criterion(y, y0)
                optimizer_t.zero_grad()
                loss.backward()
                optimizer_t.step()
                tqdm_opt.set_postfix({"loss": loss.item()})
                lr_scheduler.step()
            
        for i_p in range(len(lambda_p)):
            theta_p[i_p].requires_grad = False
        
def clear():
    if os.path.isdir("sim"):
        shutil.rmtree("sim")
    if os.path.isdir("sim"):
        print("clear error")
        sys.exit()
    for f in os.listdir("sim_data"):
        shutil.rmtree("sim_data/" + f)
        
    for f in os.listdir("save"):
        os.remove("save/" + f)
        
    for f in os.listdir("plot"):
        os.remove("plot/" + f)
      
def plot():
    min_l = torch.inf
    min_a = torch.inf
    max_l = 0
    max_a = 0
    data_it = []
    for it in pickle.load(open("save/it.pkl", "rb")):
        l = []
        a = []
        for i in it:
            d = pickle.load(open("sim_data/" + str(i) + "/data.pkl", "rb"))
            if min_l > d.y[0]:
                min_l = d.y[0]
            if min_a > d.y[1]:
                min_a = d.y[1]
            if max_l < d.y[0]:
                max_l = d.y[0]
            if max_a < d.y[1]:
                max_a = d.y[1]
            l.append(d.y[0].cpu().numpy())
            a.append(d.y[1].cpu().numpy())
            
        data_it.append([l, a])
        
    d_l = max_l - min_l
    d_a = max_a - min_a
    min_l = 0
    min_a = 0
    max_l = float(max_l + 0.05 * d_l)
    max_a = float(max_a + 0.05 * d_a)
    i = 0
    for d in data_it:
        plt.xlim([min_l, max_l])
        plt.ylim([min_a, max_a])
        plt.scatter(d[0], d[1], s = 0.5)
        plt.legend(["iteration: " + str(i + 1)])
        plt.gcf().savefig("plot/iteration" + str(i + 1) + ".png", dpi = 300)
        plt.gcf().clf()
        i += 1