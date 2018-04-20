import numpy as np
import matplotlib.pyplot as plt

out_fn = 'LOOCV_w.npz'

x, t = [], []

with open("train.txt", "r") as train:
    for line in train:
        row = line.split(',')
        x.append(float(row[0]))
        t.append(float(row[1]))

x = np.asarray(x)
t = np.asarray(t)

X = np.zeros((10, 100))

train = np.zeros((10, 99))
test = np.zeros((10, 1))


train_t = np.zeros(99)
test_t = np.zeros(1)

index = np.arange(100)
np.random.shuffle(index)
# index = index.reshape((10, 10))

# for j in range(0, 10):
#     print(index[j])

for i in range(0, 10):
    X[i, :] = np.power(x, i)

test_err = []
test_err = np.asarray(test_err)
train_err = []
train_err = np.asarray(train_err)

Eye = np.eye(10)


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

lamda = []
lamda = np.asarray(lamda)
lamda_i = 0

for step in range(0, 50):
    # lamda = np.exp(step)
    lamda = np.append(lamda, 0.1*step)
    # print(lamda)

    test_err_temp = []
    test_err_temp = np.asarray(test_err_temp)
    train_err_temp = []
    train_err_temp = np.asarray(train_err_temp)

    for k in range(0, 100):
        test_index = index[k]
        # print(test_index)
        train_index = np.delete(index, k)
        # print(train_index)

        test[:, 0] = X[:, test_index]
        test_t[0] = t[test_index]

        # print(test.shape)

        ct = 0
        for m in train_index:
            train[:, ct] = X[:, m]
            train_t[ct] = t[m]
            ct = ct + 1
        # print(train.shape)
        # print(lamda_i)
        w = np.dot(np.dot(np.linalg.inv((np.dot(train, train.T) + lamda[lamda_i]*Eye)), train), train_t)

        # print(w)

        train_err_temp = np.append(train_err_temp, rmse(np.dot(train.T, w), train_t))

        test_err_temp = np.append(test_err_temp, rmse(np.dot(test.T, w), test_t))

    lamda_i = lamda_i + 1
    # print(lamda_i)
    test_err = np.append(test_err, test_err_temp.mean())
    train_err = np.append(train_err, train_err_temp.mean())

# print(test_err)
print(np.argmin(test_err))
print(np.amin(test_err))
# print(train_err)
print(np.argmin(train_err))
print(np.amin(train_err))

lamda_min = 0.1*np.argmin(test_err)
print(lamda_min)
w_opt = np.dot(np.dot(np.linalg.inv((np.dot(X, X.T) + lamda_min*Eye)), X), t)
print(w_opt)
print(lamda_min)
plt.plot(lamda, train_err, 'b', label='train')
plt.plot(lamda, test_err, 'r', label='test')

plt.ylabel('rmse')
plt.xlabel('lambda')
plt.legend()
plt.show()

np.savez(out_fn, w_opt=w_opt, lamda_min=lamda_min, test_err_min=np.amin(test_err), train_err_min=np.amin(train_err))


exit()




