import numpy as np
import matplotlib.pyplot as plt


def serr(predictions, targets):
    return np.sum((predictions - targets) ** 2)


x, t = [], []

with open("test.txt", "r") as test:
    for line in test:
        row = line.split(',')
        x.append(float(row[0]))
        t.append(float(row[1]))

x = np.asarray(x)
t = np.asarray(t)

parameters = np.load('LOOCV_w.npz')
w = parameters['w_opt']
lamda = parameters['lamda_min']
test_err_min = parameters['test_err_min']
train_err_min = parameters['train_err_min']

print(w)
print(lamda)
print(test_err_min)
print(train_err_min)

X = np.zeros((10, len(x)))

for i in range(0, 10):
    X[i, :] = np.power(x, i)

loss = serr(np.dot(X.T, w), t)

x_plt = []
x_plt = np.asarray(x_plt)

for i in range(0, 1000):
    x_plt = np.append(x_plt, -5+i*0.01)

X_plt = np.zeros((10, len(x_plt)))

for i in range(0, 10):
    X_plt[i, :] = np.power(x_plt, i)

plt.plot(x, t, 'b', label='train set', linestyle=':')
plt.plot(x_plt, np.dot(X_plt.T, w), 'r', label='model')
plt.xlabel('x')
plt.ylabel('t')
plt.ylim((-4, 4))
plt.xlim((-6, 6))
plt.legend()
# plt.show()

print(loss)

exit()