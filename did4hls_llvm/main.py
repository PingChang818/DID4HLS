import exe
import yaml

benchmarks = ["correlation", "covariance", "gramSchmidt", "aes", "sort", "stencil"]
# !!! specify which benchmark to optimize
space = yaml.safe_load(open("data/space/" + benchmarks[0] + ".yaml", "r"))
config = yaml.safe_load(open("config.yaml", "r"))
exe.clear()
exe.dse(space, config)