import os
import psutil
import logging
import pathlib
import subprocess
import multiprocessing
from multiprocessing_logging import install_mp_handler

def run_command(command):
    try:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL)
        process.communicate(timeout = 600)
    except subprocess.TimeoutExpired:
        for child in psutil.Process(process.pid).children(recursive=True):
            child.kill()

commands = []
path = pathlib.Path().resolve().__str__().replace('\\', '/')
# !!! Specify your Vitis_HLS directory
cmd = "... .../2023.1/bin/vitis_hls -f "
for i in os.listdir("sim"):
    commands.append(cmd + path + "/sim/" + i + "/hls.tcl")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    install_mp_handler()
    pool = multiprocessing.Pool(processes = 30)
    pool.map(run_command, commands)
    pool.close()
    pool.join()