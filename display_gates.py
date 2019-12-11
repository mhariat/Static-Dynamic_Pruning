import json
import matplotlib.pyplot as plt
import numpy as np

file = '/home/marwane/PycharmProjects/experiments/stats/gate_values.json'
with open(file) as json_file:
    data = json.load(json_file)

for key in data.keys():
    data[key] = (10000 - data[key])/10000

x = list(range(len(data)))
y = list(data.values())


l = []
for key in data.keys():
    if data[key] == 0:
        l.append('')
    else:
        l.append(key)

plt.bar(x, y)
plt.xticks(x, l, rotation=20)
plt.show()