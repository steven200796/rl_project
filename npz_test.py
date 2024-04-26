import numpy as np
data = np.load("pong_run/logs/evaluations.npz")
# data = np.load("distill_run/evaluations.npz")
lst = data.files
for item in lst:
    print(item)
    print(type(data[item]))
    print(data[item])
