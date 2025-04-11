import exe
import pickle
import logging
import multiprocessing
from multiprocessing_logging import install_mp_handler

def exe_extract(graph_list):
    # !!! Specify your share folder directory
    share = "... .../share"
    space = pickle.load(open("save/space.pkl", "rb"))
    graph = exe.extract(graph_list["pragmas"], space["benchmark"], share, graph_list["tar"], False)
    pickle.dump(graph, open("graph/" + graph_list["tar"] + ".pkl", 'wb'))

if __name__ == '__main__':
    to_extract = pickle.load(open("save/to_extract.pkl", "rb"))
    avg = len(to_extract) / 20
    graph_lists = []
    last = 0.0
    ct = 0
    while last < len(to_extract):
        graph_list = {}
        graph_list["tar"] = str(ct)
        graph_list["pragmas"] = to_extract[int(last) : round(last + avg)]
        graph_lists.append(graph_list)
        last += avg
        ct += 1
        
    logging.basicConfig(level=logging.INFO)
    install_mp_handler()
    pool = multiprocessing.Pool(processes = 20)
    pool.map(exe_extract, graph_lists)
    pool.close()
    pool.join()