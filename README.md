# DID4HLS
This is the github repository for the paper "Deep Inverse Design for High-Level Synthesis" (https://doi.org/10.48550/arXiv.2407.08797)

# Environment
[Anaconda3-2023.03-0-Windows-x86_64](https://repo.anaconda.com/archive/)

[Pytorch=2.1.2 + cuda=11.8](https://pytorch.org/)

[Vitis HLS 2023.1](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis/archive-vitis.html)

The requirements for each method are contained in their respective folders.

# Experiments
Run the file main.py of each method. Move the simulation results in sim_data folder of each method to did4hls/result and run adrs.py in did4hls to generate the Pareto comparison.

# Acknowledgements
We made modifications to the open-sourced code of [GRASP5](https://github.com/nibst/GRASP_DSE), and re-implemented the remaining baselines as closely as possible according to their respective descriptions.
