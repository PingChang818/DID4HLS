import copy
import yaml
import utils
import pickle
import logging
import numpy as np
import multiprocessing
from multiprocessing_logging import install_mp_handler

def s_a(sample, temp, lambda_p, model_l, model_a, space, dt, share, tar):
    pragma = copy.deepcopy(sample)
    f_1 = utils.extract([pragma], space, share, tar)[0]
    while len(f_1) < 6:
        f_1 = utils.extract([pragma], space, share, tar)[0]
        open("sa_log/" + tar + ".txt", "a").write("reparse\n")
    changed = False
    while not changed:
        if "ii" in pragma.keys():
            ii = pragma["ii"]
            unroll = pragma["unroll"]
            sel_loop = np.random.choice(list(range(len(ii))))
            if ii[sel_loop] != -1:
                if np.random.rand() > 0.5:
                    changed = True
                    idx = space["ii"][sel_loop].index(ii[sel_loop])
                    if idx == 0:
                        ii[sel_loop] = space["ii"][sel_loop][1]
                    elif idx == len(space["ii"][sel_loop]) - 1:
                        ii[sel_loop] = space["ii"][sel_loop][idx - 1]
                    elif np.random.rand() > 0.5:
                        ii[sel_loop] = space["ii"][sel_loop][idx + 1]
                    else:
                        ii[sel_loop] = space["ii"][sel_loop][idx - 1]
                else:
                    if len(space["unroll"][sel_loop]) > 1:
                        changed = True
                        idx = space["unroll"][sel_loop].index(unroll[sel_loop])
                        if idx == 0:
                            unroll[sel_loop] = space["unroll"][sel_loop][1]
                        elif idx == len(space["unroll"][sel_loop]) - 1:
                            unroll[sel_loop] = space["unroll"][sel_loop][idx - 1]
                        elif np.random.rand() > 0.5:
                            unroll[sel_loop] = space["unroll"][sel_loop][idx + 1]
                        else:
                            unroll[sel_loop] = space["unroll"][sel_loop][idx - 1]
            if not changed and len(space["ii"][sel_loop]) > 0 and np.random.rand() > 0.5:
                pointer = sel_loop
                while pointer != -1:
                    pointer = space["nest"][pointer]
                    
                if pointer == sel_loop:
                    changed = True
                    ii[sel_loop] = space["ii"][sel_loop][0]
                    pointer += 1
                    while pointer < len(space["nest"]) and space["nest"][pointer] != -1:
                        ii[pointer] = -1
                        unroll[pointer] = -1
                        pointer += 1
                        
                elif unroll[sel_loop] == -1:
                    changed = True
                    ii[pointer] = -1
                    pointer += 1
                    while space["nest"][pointer] != -1:
                        ii[pointer] = -1
                        unroll[pointer] = space["bound"][pointer]
                        pointer += 1
                        
                elif len(space["unroll"][sel_loop]) > 1:
                    changed = True
                    idx = space["unroll"][sel_loop].index(unroll[sel_loop])
                    if idx == 0:
                        unroll[sel_loop] = space["unroll"][sel_loop][1]
                    elif idx == len(space["unroll"][sel_loop]) - 1:
                        unroll[sel_loop] = space["unroll"][sel_loop][idx - 1]
                    elif np.random.rand() > 0.5:
                        unroll[sel_loop] = space["unroll"][sel_loop][idx + 1]
                    else:
                        unroll[sel_loop] = space["unroll"][sel_loop][idx - 1]
            if not changed and len(space["ii"][sel_loop]) == 0 and np.random.rand() > 0.5:
                if len(space["unroll"][sel_loop]) > 1:
                    changed = True
                    idx = space["unroll"][sel_loop].index(unroll[sel_loop])
                    if idx == 0:
                        unroll[sel_loop] = space["unroll"][sel_loop][1]
                    elif idx == len(space["unroll"][sel_loop]) - 1:
                        unroll[sel_loop] = space["unroll"][sel_loop][idx - 1]
                    elif np.random.rand() > 0.5:
                        unroll[sel_loop] = space["unroll"][sel_loop][idx + 1]
                    else:
                        unroll[sel_loop] = space["unroll"][sel_loop][idx - 1]
            if changed:
                pragma["ii"] = ii
                pragma["unroll"] = unroll
        if not changed and "interface" in pragma.keys() and len(space["interface"]) > 1 and np.random.rand() > 0.5:
            var = np.random.choice(space["interface"])
            interface = pragma["interface"][var]
            p_itf = space["p_itf"][space["interface"].index(var)]
            if len(p_itf) > 1:
                changed = True
                if p_itf.index(interface[1]) == 0:
                    interface[1] = p_itf[1]
                elif p_itf.index(interface[1]) == len(p_itf) - 1:
                    interface[1] = p_itf[-2]
                elif np.random.rand() > 0.5:
                    interface[1] = p_itf[p_itf.index(interface[1]) + 1]
                else:
                    interface[1] = p_itf[p_itf.index(interface[1]) - 1]
                pragma["interface"][var] = interface
        if not changed and "inline" in pragma.keys() and np.random.rand() > 0.5:
            changed = True
            fc = np.random.choice(space["function"])
            if pragma["inline"][fc]:
                pragma["inline"][fc] = False
            else:
                pragma["inline"][fc] = True
                
    t_1 = np.zeros(len(dt) + 4, dtype = int)
    t_1[-4] = f_1[0]
    t_1[-3] = f_1[1]
    t_1[-2] = f_1[2]
    t_1[-1] = f_1[3]
    for i in range(len(f_1[4])):
        if isinstance(np.where(dt == f_1[4][i])[0], list):
            t_1[np.where(dt == f_1[4][i])[0][0]] = f_1[5][i]
        else:
            t_1[np.where(dt == f_1[4][i])[0]] = f_1[5][i]
        
    f_2 = utils.extract([pragma], space, share, tar)[0]
    while len(f_2) < 6:
        open("sa_log/" + tar + ".txt", "a").write("reparse\n")
        f_2 = utils.extract([pragma], space, share, tar)[0]
        
    t_2 = np.zeros(len(dt) + 4, dtype = int)
    t_2[-4] = f_2[0]
    t_2[-3] = f_2[1]
    t_2[-2] = f_2[2]
    t_2[-1] = f_2[3]
    for i in range(len(f_2[4])):
        if isinstance(np.where(dt == f_2[4][i])[0], list):
            t_2[np.where(dt == f_2[4][i])[0][0]] = f_2[5][i]
        else:
            t_2[np.where(dt == f_2[4][i])[0]] = f_2[5][i]
    
    t_1 = t_1.reshape((1,len(t_1)))
    t_2 = t_2.reshape((1,len(t_2)))
    c_1 = lambda_p * model_l.predict(t_1)[0] + (1 - lambda_p) * model_a.predict(t_1)[0]
    c_2 = lambda_p * model_l.predict(t_2)[0] + (1 - lambda_p) * model_a.predict(t_2)[0]
    delta = c_2 - c_1
    if delta < 0 or np.random.rand() < np.exp(-1 * delta / temp):
        return pragma, 0
    else:
        return sample, 1
    
def exe(lambda_p):
    r = 500
    nt = 50
    temp = 10
    temp_stop = 1e-7
    gama = 0.1
    # !!! Specify your share folder directory
    share = "... .../share"
    tar = str(int(lambda_p * 100))
    dt = pickle.load(open("save/dt.pkl", "rb"))
    space = pickle.load(open("save/space.pkl", "rb"))
    model_l, model_a = pickle.load(open("save/model.pkl", "rb"))
    theta, info = utils.theta_init(space)
    prob = utils.prob(theta, info).cpu().numpy()
    sample = utils.sample(prob, info, space, 1)[0]
    rt = 0
    while rt < r and round(temp, 8) > temp_stop:
        it = 0
        while it < nt:
            open("sa_log/" + tar + ".txt", "a").write("it: {}, temp: {:.2E}, rt: {}\n".format(it, temp, rt))
            try:
                sample, rej = s_a(sample, temp, lambda_p, model_l, model_a, space, dt, share, tar)
            except Exception as e:
                yaml.dump(sample, open("sa/" + tar + ".yaml", "w"))
                open("sa_log/" + tar + "_err.txt", "a").write(str(e) + "\n")
                return
            rt += rej
            it += 1
            
        temp *= gama
        
    yaml.dump(sample, open("sa/" + tar + ".yaml", "w"))
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    install_mp_handler()
    for i in range(3):
        pool = multiprocessing.Pool(processes = 20)
        pool.map(exe, np.linspace(0, 1, 30).tolist()[i * 10 : (i + 1) * 10])
        pool.close()
        pool.join()