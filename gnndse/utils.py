import os
import sys
import json
import copy
import yaml
import torch
import shutil
import pickle
import pathlib
import paramiko
import numpy as np

def cdfg(tar):
    # !!! Type in the user name, host IP, and login password of your Linux
    user = '... ...'
    host = '... ...'
    password = '... ...'
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password)
    stdin, stdout, stderr = client.exec_command('python3 /mnt/hgfs/share/cdfg.py ' + "gnn_dse_" + tar)
    stdout.read().decode()
    client.close()

def graph(cdfg, is_train = True):
    dn = []
    det = []
    ded = []
    emb = []
    if os.path.isfile("emb.pkl"):
        emb = pickle.load(open("emb.pkl", "rb"))
    for k, d in cdfg._node.items():
        if is_train and not d["text"] in emb:
            emb.append(d["text"])
        if d["text"] in emb:
            dn.append(emb.index(d["text"]))
        else:
            dn.append(0)
        
    for k, d in cdfg._succ.items():
        for ke, de in d.items():
            flow = "flow: " + str(de[0]["flow"])
            if is_train and not flow in emb:
                emb.append(flow)
            if flow in emb:
                det.append(emb.index(flow))
            else:
                det.append(0)
            ded.append([k, ke])
            
    if len(emb) > 128:
        print("embedding error")
        sys.exit()
    if is_train:
        pickle.dump(emb, open("emb.pkl", "wb"))
        
    return dn, det, ded

def code4clang(pragma, benchmark, order, share, tar):
    code = open("data/benchmark/" + benchmark + ".cpp", "r").readlines()
    if "inline" in pragma.keys():
        for fc in pragma["inline"].keys():
            n_lines = len(code)
            if pragma["inline"][fc]:
                found = False
                for i_line in range(n_lines):
                    if fc + " (" in code[i_line]:
                        found = True
                        for j in range(len(code[i_line])):
                            if code[i_line][j] != " ":
                                code[i_line] = code[i_line][0 : j] + "inline " + code[i_line][j :]
                                code.insert(i_line, "#pragma clang optimize on\n")
                                for j_line in range(i_line, n_lines):
                                    if code[j_line].replace(" ", "")[0] == "}":
                                        code.insert(j_line + 1, "#pragma clang optimize off\n")
                                        break
                                    
                                break
                            
                    if found:
                        break
                        
    if "unroll" in pragma.keys():
        for line in range(len(code) - 1, -1, -1):
            if "loop_" in code[line]:
                li = int(code[line].split("_")[1].split(":")[0])
                si = code[line].find("loop_")
                ei = code[line].find(": ") + 2
                code[line] = code[line][0 : si] + code[line][ei :]
                if pragma["unroll"][li] != -1:
                    code.insert(line, "#pragma clang loop unroll_count(" + str(pragma["unroll"][li]) +")\n")
                if pragma["ii"][li] != -1:
                    code.insert(line, "#pragma clang loop pipeline_initiation_interval(" + str(pragma["ii"][li]) + ")\n")
                
    open(share + "/code_gnn_dse_" + tar + "/" + order + ".cpp", "w").write("".join(code))

def theta_init(space, prob_single = False, pragma = None):
    info = {}
    tensor = torch.tensor([])
    keys = space.keys()
    if "inline" in keys:
        info["inline"] = {}
        for i_il, il in enumerate(space["inline"]):
            prob = None
            if prob_single:
                if pragma["inline"][space["function"][i_il]]:
                    prob = torch.tensor([1])
                else:
                    prob = torch.tensor([0])
            else:
                prob = torch.logit(torch.tensor([il]))
            tensor = torch.cat([tensor, prob])
            info["inline"][space["function"][i_il]] = len(tensor) - 1
            
    if "interface" in keys:
        info["interface"] = {}
        p_itf = space["p_itf"]
        for i_itf, itf in enumerate(space["interface"]):
            tensor_interface = torch.zeros(len(p_itf[i_itf]) + 1)
            if prob_single:
                if pragma["interface"][itf][0] == "block":
                    tensor_interface[0] = 1
                else:
                    tensor_interface[0] = 0
                tensor_interface[p_itf[i_itf].index(pragma["interface"][itf][1]) + 1] = 1
            start = len(tensor)
            tensor = torch.cat([tensor, tensor_interface])
            info["interface"][itf] = [start, len(tensor)]
            
    if "pipeline" in keys:
        info_loops = {}
        pipeline = space["pipeline"]
        ii = space["ii"]
        unroll = space["unroll"]
        for i_loop in range(len(pipeline)):
            info_loop = {}
            loop_pipeline = torch.tensor([])
            if pipeline[i_loop] > 0:
                if prob_single:
                    if pragma["ii"][i_loop] == -1:
                        loop_pipeline = torch.tensor([0])
                    else:
                        loop_pipeline = torch.tensor([1])
                else:
                    loop_pipeline = torch.logit(torch.tensor([pipeline[i_loop]]))
                len_ii = len(ii[i_loop])
                if len_ii > 1:
                    loop_pipeline = torch.cat([loop_pipeline, torch.zeros(len_ii)])
                    if prob_single and pragma["ii"][i_loop] != -1:
                        loop_pipeline[ii[i_loop].index(pragma["ii"][i_loop]) + 1] = 1
                start = len(tensor)
                tensor = torch.cat([tensor, loop_pipeline])
                info_loop["pipeline"] = [start, len(tensor)]
            len_unroll = len(unroll[i_loop])
            if len_unroll > 1:
                loop_unroll = torch.zeros(len_unroll)
                if prob_single:
                    if pragma["unroll"][i_loop] == -1:
                        loop_unroll[-1] = 1
                    else:
                        loop_unroll[unroll[i_loop].index(pragma["unroll"][i_loop])] = 1
                start = len(tensor)
                tensor = torch.cat([tensor, loop_unroll])
                info_loop["unroll"] = [start, len(tensor)]
            if info_loop != {}:
                info_loops["loop_" + str(i_loop)] = info_loop
                
        if info_loops != {}:
            info["loop"] = info_loops
            
    return tensor, info

def prob(prob, info):
    keys = info.keys()
    if "inline" in keys:
        inline = info["inline"]
        for var in inline.keys():
            prob[inline[var]] = torch.sigmoid(prob[inline[var]])
            
    if "interface" in keys:
        interface = info["interface"]
        for itf in interface.keys():
            prob[interface[itf][0]] = torch.sigmoid(prob[interface[itf][0]])
            prob[interface[itf][0] + 1 : interface[itf][1]] = torch.softmax(prob[interface[itf][0] + 1 : interface[itf][1]], 0)
            
    if "loop" in keys:
        info_loops = info["loop"]
        for key_loop in info_loops.keys():
            if "pipeline" in info_loops[key_loop].keys():
                pipeline = info_loops[key_loop]["pipeline"]
                prob[pipeline[0]] = torch.sigmoid(prob[pipeline[0]])
                if pipeline[1] - pipeline[0] > 1:
                    prob[pipeline[0] + 1 : pipeline[1]] = torch.softmax(prob[pipeline[0] + 1 : pipeline[1]], 0)
            if "unroll" in info_loops[key_loop].keys():
                unroll = info_loops[key_loop]["unroll"]
                prob[unroll[0] : unroll[1]] = torch.softmax(prob[unroll[0] : unroll[1]], 0)
                
    return prob

def sample(prob, info, space, iteration_size):
    keys = info.keys()
    samples_pragma = []
    for i in range(iteration_size):
        samples_pragma.append(dict())
        
    samples_inline_pragma = []
    if "inline" in keys:
        for i in range(iteration_size):
            sample_inline_pragma = {}
            for fc in info["inline"].keys():
                prob_inline = prob[info["inline"][fc]]
                sample_inline_pragma[fc] = np.random.choice([True, False], p = [prob_inline, 1 - prob_inline])
                
            samples_inline_pragma.append(sample_inline_pragma)
        
    samples_interface_pragma = []
    if "interface" in keys:
        for i in range(iteration_size):
            sample_interface_pragma = {}
            interface = space["interface"]
            p_itf = space["p_itf"]
            for i_itf, itf in enumerate(interface):
                prob_itf = prob[info["interface"][itf][0] : info["interface"][itf][1]]
                factor = np.random.choice(p_itf[i_itf], p = prob_itf[1 :])
                type_partition = "block"
                if factor != 1:
                    type_partition = str(np.random.choice(["block", "cyclic"], p = [prob_itf[0], 1 - prob_itf[0]]))
                sample_interface_pragma[itf] = [type_partition, factor]
                
            samples_interface_pragma.append(sample_interface_pragma)
            
    samples_loop_pragma = []
    if "loop" in keys:
        nest = space["nest"]
        pipeline = space["pipeline"]
        ii = space["ii"]
        unroll = space["unroll"]
        bound = space["bound"]
        n_loop = len(nest)
        for i in range(iteration_size):
            ii_sample = [-1] * n_loop
            unroll_sample = [-1] * n_loop
            for i_loop in range(n_loop):
                above_pipelined = False
                above = nest[i_loop]
                while above != -1:
                    if ii_sample[above] != -1:
                        above_pipelined = True
                        break
                    above = nest[above]
                    
                if above_pipelined:
                    if unroll[i_loop][-1] != bound[i_loop]:
                        print("Space error: the max factor and bound of the pipelined loop are not same!")
                        sys.exit()
                    unroll_sample[i_loop] = -1
                else:
                    if len(unroll[i_loop]) > 1:
                        loop_unroll = info["loop"]["loop_" + str(i_loop)]["unroll"]
                        prob_loop_unroll = prob[loop_unroll[0] : loop_unroll[1]]
                    if pipeline[i_loop] > 0:
                        loop_pipeline = info["loop"]["loop_" + str(i_loop)]["pipeline"]
                        prob_loop_pipeline = prob[loop_pipeline[0] : loop_pipeline[1]]
                        pipelined = np.random.choice([True, False], p = [prob_loop_pipeline[0], 1 - prob_loop_pipeline[0]])
                        if pipelined:
                            if len(ii[i_loop]) > 1:
                                ii_sample[i_loop] = np.random.choice(ii[i_loop], p = prob_loop_pipeline[1 :])
                            else:
                                ii_sample[i_loop] = ii[i_loop][0]
                            if len(unroll[i_loop]) > 1:
                                if unroll[i_loop][-1] == bound[i_loop]:
                                    unroll_sample[i_loop] = np.random.choice(unroll[i_loop], p = prob_loop_unroll)
                                    while unroll_sample[i_loop] == bound[i_loop]:
                                        unroll_sample[i_loop] = np.random.choice(unroll[i_loop], p = prob_loop_unroll)
                                        
                                else:
                                    unroll_sample[i_loop] = np.random.choice(unroll[i_loop], p = prob_loop_unroll)
                            else:
                                unroll_sample[i_loop] = unroll[i_loop][0]
                    if pipeline[i_loop] == 0 or not pipelined:
                        if len(unroll[i_loop]) > 1:
                            unroll_sample[i_loop] = np.random.choice(unroll[i_loop], p = prob_loop_unroll)
                        else:
                            unroll_sample[i_loop] = unroll[i_loop][0]
                            
            samples_loop_pragma.append({"ii": ii_sample, "unroll": unroll_sample})
            
    if len(samples_inline_pragma) > 0:
        for i in range(iteration_size):
            samples_pragma[i].update({"inline": samples_inline_pragma[i]})
            
    if len(samples_interface_pragma) > 0:
        for i in range(iteration_size):
            samples_pragma[i].update({"interface": samples_interface_pragma[i]})
            
    if len(samples_loop_pragma) > 0:
        for i in range(iteration_size):
            samples_pragma[i].update(samples_loop_pragma[i])
    
    return samples_pragma

def sim(samples_pragma, space):
    simulated_id = []
    simulated_pragma = []
    for i in os.listdir("sim_data"):
        simulated_id.append(int(i))
        simulated_pragma.append(yaml.load(open("sim_data/" + i + "/pragma.yaml", "r"), Loader=yaml.Loader))
        
    samples_new = []
    for sample_pragma in samples_pragma:
        if not sample_pragma in simulated_pragma:
            samples_new.append(sample_pragma)
            
    new_id = []
    new_unique = []
    start = -1
    if len(simulated_id) > 0:
        start = max(simulated_id)
    order = copy.deepcopy(start)
    samples_tmp = copy.deepcopy(samples_new)
    while len(samples_tmp) > 0:
        pragma = samples_tmp.pop(0)
        if pragma not in samples_tmp:
            order += 1
            new_id.append(order)
            new_unique.append(pragma)
            
    os.mkdir("sim")
    keys = space.keys()
    benchmark = space["benchmark"]
    benchmark_code = open("data/benchmark/" + benchmark + ".cpp", "r").readlines()
    for i in range(len(new_id)):
        id_str = str(new_id[i])
        os.mkdir("sim/" + id_str)
        pragma = new_unique[i]
        yaml.dump(pragma, open("sim/" + id_str + "/pragma.yaml", "w"))
        code = benchmark_code.copy()
        if "interface" in keys:
            n_lines = len(code)
            found = False
            for i_line in range(n_lines):
                if benchmark + " (" in code[i_line]:
                    for j_line in range(i_line, n_lines):
                        if code[j_line][-2] == "{":
                            interface = pragma["interface"]
                            for key in interface.keys():
                                code.insert(j_line + 1,
                                            "#pragma HLS ARRAY_PARTITION dim=1 factor=" +
                                            str(interface[key][1]) +
                                            " type=" + interface[key][0] +
                                            " variable=" + key + "\n")
                                
                            found = True
                            break
                        
                    if found:
                        break
        
        if "inline" in keys:
            for fc in pragma["inline"].keys():
                n_lines = len(code)
                il = " off"
                if pragma["inline"][fc]:
                    il = ""
                found = False
                for i_line in range(n_lines):
                    if fc + " (" in code[i_line]:
                        for j_line in range(i_line, n_lines):
                            if code[j_line][-2] == "{":
                                code.insert(j_line + 1, "#pragma HLS INLINE" + il + "\n")
                                found = True
                                break
                            
                        if found:
                            break
                                
        if "pipeline" in keys:
            n_lines = len(code)
            ii = pragma["ii"]
            unroll = pragma["unroll"]
            for i_line in range(n_lines - 1, -1, -1):
                if "loop_" in code[i_line]:
                    i_loop = int(code[i_line].split("_")[1].split(":")[0])
                    if unroll[i_loop] != -1:
                        code.insert(i_line + 1, "#pragma HLS UNROLL factor=" + str(unroll[i_loop]) +"\n")
                    if ii[i_loop] != -1:
                        code.insert(i_line + 1, "#pragma HLS PIPELINE II=" + str(ii[i_loop]) +"\n")
                        
        open("sim/" + id_str + "/" + benchmark + ".cpp", "w").write("".join(code))
        for file in os.listdir("data/benchmark/"):
            if file == benchmark + ".h":
                shutil.copy("data/benchmark/" + benchmark + ".h", "sim/" + id_str)
                break
                
        path = pathlib.Path().resolve().__str__().replace('\\', '/')
        tcl = (
            'cd ' + path + '/sim\n' +
            'open_project ' + id_str + '\n' +
            'cd ' + id_str + '\n' +
            'open_solution solution\n' +
            'config_array_partition -complete_threshold 0\n'
            'config_array_partition -throughput_driven off\n'
            'config_compile -pipeline_loops 0\n' +
            'config_compile -pipeline_style stp\n' +
            'config_compile -enable_auto_rewind=0\n' +
            'config_compile -pipeline_flush_in_task never\n' +
            'config_unroll -tripcount_threshold 0\n' +
            'add_files ' + path + '/sim/' + id_str + "/" + benchmark + '.cpp\n' +
            'set_top ' + benchmark +'\n' +
            'create_clock -period ' + str(space["clock"][0]) +'\n' +
            'set_part xq7vx690t-rf1930-1I\n' +
            'csynth_design\n' +
            'exit')
        open("sim/" + id_str + "/hls.tcl", "w").write(tcl)
        
    new_id_valid = []
    if len(new_id) > 0:
        print("synthesizing {} designs...".format(len(new_id)))
        os.system('cmd /c "python hls.py"')
        print("generating...")
        for i in new_id:
            id_str = str(i)
            if os.path.exists("sim/" + id_str + "/solution/solution_data.json"):
                rpt = json.load(open("sim/" + id_str + "/solution/solution_data.json", "r"))
                lw = rpt["ModuleInfo"]["Metrics"][space["benchmark"]]["Latency"]["LatencyWorst"]
                if lw != '':
                    start += 1
                    y_l = int(lw)
                    y_a = 0
                    area = rpt["ModuleInfo"]["Metrics"][space["benchmark"]]["Area"]
                    y_a += float(area["FF"]) / float(area["AVAIL_FF"])
                    y_a += float(area["LUT"]) / float(area["AVAIL_LUT"])
                    y_a += float(area["BRAM_18K"]) / float(area["AVAIL_BRAM"])
                    y_a += float(area["DSP"]) / float(area["AVAIL_DSP"])
                    y_a /= 4
                    new_id_valid.append(start)
                    os.mkdir("sim_data/" + str(start))
                    pickle.dump([y_l, y_a], open("sim_data/" + str(start) + "/data.pkl", "wb"))
                    shutil.copy("sim/" + id_str + "/pragma.yaml", "sim_data/" + str(start))
        
    shutil.rmtree("sim")
    
    return new_id_valid