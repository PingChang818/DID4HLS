import os
import sys
import pickle
import logging
import multiprocessing
import xml.etree.ElementTree as ET
from multiprocessing_logging import install_mp_handler

def cdfg(point):
    hls_path = "sim/" + point + "/solution/.autopilot/db/"
    module = pickle.load(open("module.pkl", "rb"))
    xml = [module]
    for file in os.listdir(hls_path):
        name = file.split(".")
        if len(name) == 2 and name[0] != module and name[1] == "adb":
            xml.append(name[0])
            
    if not os.path.exists(hls_path + module + ".adb"):
        print("syn error")
    else:
        nodes = []
        edges = []
        calls = []
        returns = []
        nid = -1
        eid = -1
        for file in xml:
            tree = ET.parse(hls_path + file + ".adb")
            nodes_current = []
            edges_current = []
            if tree.getroot()[0][2].tag != "cdfg":
                print("error: cdfg")
                sys.exit()
            if tree.getroot()[0][2][3].tag != "ports":
                print("error: ports")
                sys.exit()
            for item in tree.getroot()[0][2][3]:
                if item.tag == "item":
                    nid += 1
                    node = dict()
                    node["id"] = nid
                    node["oid"] = int(item[0][0][1].text)
                    node["type"] = [int(item[0][0][0].text), int(item[1].text)]
                    node["file"] = file
                    node["name"] = item[0][0][2].text
                    node["line"] = int(item[0][0][5].text)
                    node["bitwidth"] = int(item[0][1].text)
                    nodes_current.append(node)
            
            if tree.getroot()[0][2][4].tag != "nodes":
                print("error: nodes")
                sys.exit()
            for item in tree.getroot()[0][2][4]:
                if item.tag == "item":
                    nid += 1
                    node = dict()
                    node["id"] = nid
                    node["oid"] = int(item[0][0][1].text)
                    node["type"] = [int(item[0][0][0].text)]
                    node["file"] = file
                    node["name"] = item[0][0][2].text
                    node["line"] = int(item[0][0][5].text)
                    node["bitwidth"] = int(item[0][1].text)
                    node["opcode"] = item[2].text
                    es = []
                    for edge in item[1]:
                        if edge.tag == "item":
                            es.append(int(edge.text))
                            
                    node["edges"] = es
                    nodes_current.append(node)
                    if file != module and node["opcode"] == "br":
                        returns.append(nid)
            
            if tree.getroot()[0][2][5].tag != "consts":
                print("error: consts")
                sys.exit()
            for item in tree.getroot()[0][2][5]:
                if item.tag == "item":
                    nid += 1
                    node = dict()
                    node["id"] = nid
                    node["oid"] = int(item[0][0][1].text)
                    node["type"] = [int(item[0][0][0].text), int(item[1].text)]
                    node["file"] = file
                    node["name"] = item[0][0][2].text
                    node["line"] = int(item[0][0][5].text)
                    node["bitwidth"] = int(item[0][1].text)
                    nodes_current.append(node)
                    if node["name"] in xml:
                        calls.append(nid)
            
            ct = 0
            if tree.getroot()[0][2][6].tag != "blocks":
                print("error: blocks")
                sys.exit()
            for item in tree.getroot()[0][2][6]:
                if item.tag == "item":
                    ct -= 1
                    nid += 1
                    node = dict()
                    node["id"] = nid
                    node["oid"] = int(item[0][1].text)
                    node["type"] = [int(item[0][0].text), 0]
                    node["file"] = file
                    node["name"] = item[0][2].text
                    node["line"] = int(item[0][5].text)
                    node["bitwidth"] = 0
                    bn = []
                    for n in item[1]:
                        if n.tag == "item":
                            bn.append(int(n.text))
                            
                    node["nodes"] = bn
                    nodes_current.append(node)
                    
            while ct < 0:
                f = nodes_current[ct]
                for bn in f["nodes"]:
                    find = False
                    for n in nodes_current:
                        if n["oid"] == bn:
                            eid += 1
                            edge = dict()
                            edge["id"] = eid
                            edge["type"] = [4, 50]
                            edge["direction"] = [f["id"], n["id"]]
                            edge["file"] = file
                            edges_current.append(edge)
                            ct += 1
                            find = True
                            break
                        
                    if not find:
                        print("error: find_1")
                        sys.exit()
            
            if tree.getroot()[0][2][7].tag != "edges":
                print("error: edges")
                sys.exit()
            for item in tree.getroot()[0][2][7]:
                if item.tag == "item":
                    eid += 1
                    edge = dict()
                    edge["id"] = eid
                    edge["oid"] = int(item[0].text)
                    edge["type"] = [4, int(item[1].text)]
                    edge["file"] = file
                    edge["direction"] = [None, None]
                    f = int(item[2].text)
                    t = int(item[3].text)
                    for n in nodes_current:
                        if n["oid"] == f:
                            edge["direction"][0] = n["id"]
                        if n["oid"] == t:
                            edge["direction"][1] = n["id"]
                        if edge["direction"][0] is not None and edge["direction"][1] is not None:
                            edges_current.append(edge)
                            break
                    
            nodes += nodes_current
            edges += edges_current
            
        for e in edges:
            if e["direction"][0] is None or e["direction"][1] is None:
                print("direction error")
                sys.exit()
            
        for c in calls:
            for r in returns:
                if nodes[r]["file"] == nodes[c]["name"]:
                    eid += 1
                    edge = dict()
                    edge["id"] = eid
                    edge["type"] = [4, 60]
                    edge["direction"] = [nodes[r]["id"], nodes[c]["id"]]
                    edges.append(edge)
                    
        if nid < 0 or eid < 0:
            print("error: cdfg")
            sys.exit()
        pickle.dump([nodes, edges], open("sim/" + point + "/cdfg.pkl", 'wb'))
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    install_mp_handler()
    pool = multiprocessing.Pool(processes = multiprocessing.cpu_count())
    pool.map(cdfg, pickle.load(open("new.pkl", "rb")))
