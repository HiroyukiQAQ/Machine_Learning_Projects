import numpy as np
import matplotlib.pyplot as plt

x, t = [], []

with open("test.txt", "r") as train:
    for line in train:
        row = line.split(',')
        x.append(float(row[0]))
        t.append(float(row[1]))

x = np.asarray(x)
t = np.asarray(t)

plt.plot(x, t, 'b', label='test_data', linestyle=':')
plt.legend()
plt.xlabel('x')
plt.ylabel('t')
plt.xlim((-6, 6))
plt.ylim((-4, 4))
plt.show()

exit()