import exe
import yaml
import datetime

start = datetime.datetime.now()
benchmarks = ["correlation", "covariance", "gramSchmidt", "aes", "sort", "stencil"]
# !!! Specify which benchmark to optimize
space = yaml.safe_load(open("data/space/" + benchmarks[0] + ".yaml", "r"))
config = yaml.safe_load(open("config.yaml", "r"))
exe.clear()
exe.dse(space, config)
end = datetime.datetime.now()
print("time:")
print((end-start).seconds)