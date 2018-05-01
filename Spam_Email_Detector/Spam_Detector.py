import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import matplotlib.pyplot as plt

np.seterr(all='ignore')

# Read the train.csv
with open("train.csv", "r", encoding='latin1') as train:
    csvfile = pd.read_csv(train, header=None)

csvfile = csvfile.values.tolist()

labeltxt = []
smstxt = []

for line in csvfile:
    labeltxt.append(line[0])
    smstxt.append(line[1])

labelnum = np.array([])

for i in labeltxt:
    if i == 'ham':
        labelnum = np.append(labelnum, 1)
    if i == 'spam':
        labelnum = np.append(labelnum, 0)

vectorizer = CountVectorizer()

smsnum = vectorizer.fit_transform(smstxt)

smsnum = smsnum.toarray()

transformer = TfidfTransformer(smooth_idf=True)

smstf = transformer.fit_transform(smsnum).toarray()


# generate a random value w0 from range(-1,6), as the initial weights
w0 = np.random.randint(-1, 7, 6277, dtype=int)


# Computation Functions
def sigmoid(w, x):
    return 1/(1+np.exp(-(np.dot(x, w)), dtype=np.float128))


def gradient_sigmoid(w, x):
    return ((1/(2+np.exp(np.dot(x, w), dtype=np.float128) + np.exp(-np.dot(x, w), dtype=np.float128)))
            .reshape(len(x), 1))*x


def loss(y, x, w, lam):
    loss_compute = ((-y)*(np.log(sigmoid(w, x), dtype=np.float128))) \
                   - ((1-y)*(np.log(1-sigmoid(w, x), dtype=np.float128))) \
                   + (lam*np.power(np.linalg.norm(w), 2))
    return sum(loss_compute)


def gradient(y, x, w, lam):
    y_col = y.reshape(1, len(y))
    exp = (np.exp(-np.dot(x, w))).reshape(1, len(y))
    g = np.dot((-y_col)*(1+exp), gradient_sigmoid(w, x)) + np.dot((1-y_col)*((1+exp)/exp), gradient_sigmoid(w, x))
    g = (g.reshape(len(w),)) + 2*lam*w
    return g


def gradient_descent(w, x, y, lam):
    w_old = w
    step = 0.5
    loss_old = loss(y, x, w_old, lam)
    w_new = w_old - step*gradient(y, x, w_old, lam)
    loss_new = loss(y, x, w_new, lam)

    if loss_new - loss_old >= 0:
        return w_old
    else:
        return gradient_descent(w_new, x, y, lam)


# 6-fold cross validation
index = np.arange(3000)
np.random.shuffle(index)
index = index.reshape(6, 500)

test = np.zeros((500, 6277), dtype=np.float128)
train = np.zeros((2500, 6277), dtype=np.float128)

test_label = np.zeros(500, dtype=int)
train_label = np.zeros(2500, dtype=int)

lamda = np.array([])
rate_test = np.array([])
rate_train = np.array([])

for i in range(0, 20):

    lamda = np.append(lamda, 0.1+i*0.01)

    rate_temp_test = 0
    rate_temp_train = 0

    for k in range(0, 6):
        test_index = index[k]
        train_index = np.delete(index, k, 0).reshape(2500)

        ct = 0
        for n in test_index:
            test[ct] = smstf[n]
            test_label[ct] = labelnum[n]
            ct = ct + 1

        ct = 0
        for n in train_index:
            train[ct] = smstf[n]
            train_label[ct] = labelnum[n]
            ct = ct + 1

        w_temp = gradient_descent(w0, train, train_label, lamda[i])

        # compute accuracy on test set
        hit = 0
        j = 0
        for sms in test:
            result = sigmoid(w_temp, sms)
            if result >= 0.5:
                result = 1
            else:
                result = 0

            if result == test_label[j]:
                hit = hit + 1
            j = j + 1

        rate_temp_test = rate_temp_test + hit/500

        # compute accuracy on train set
        hit = 0
        j = 0
        for sms in train:
            result = sigmoid(w_temp, sms)
            if result >= 0.5:
                result = 1
            else:
                result = 0

            if result == train_label[j]:
                hit = hit + 1
            j = j + 1

        rate_temp_train = rate_temp_train + hit/2500

    rate_test = np.append(rate_test, 100*(rate_temp_test/6))
    rate_train = np.append(rate_train, 100*(rate_temp_train/6))

print('optimal average accuracy rate on test: ', np.argmax(rate_test), np.amax(rate_test))
lamda_opt = lamda[np.argmax(rate_test)]
print('optimal lambda is：', lamda_opt)

w_opt = gradient_descent(w0, smstf, labelnum, lamda_opt)
np.savez('w.npz', w_opt=w_opt)

print('optimal loss is: ', loss(labelnum, smstf, w_opt, lamda_opt))

plt.plot(lamda, rate_train, 'b', label='trainset accuracy')
plt.plot(lamda, rate_test, 'r', label='testset accuracy')

plt.ylabel('accuracy rate')
plt.xlabel('lambda')
plt.legend()
plt.show()

exit()