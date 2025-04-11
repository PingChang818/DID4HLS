import exe
import yaml

benchmarks = ["correlation", "covariance", "gramSchmidt", "aes", "sort", "stencil"]
# !!! Specify which benchmark to optimize
space = yaml.safe_load(open("data/space/" + benchmarks[0] + ".yaml", "r"))
config = yaml.safe_load(open("config.yaml", "r"))
config_DM = yaml.safe_load(open("config_DM.yaml", "r"))
exe.clear()
exe.dse(space, config, config_DM)