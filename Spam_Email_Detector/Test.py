import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd

parameters = np.load('w.npz')
w_opt = parameters['w_opt']
print(w_opt.shape)
# open trainset
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


# open testset
def sigmoid(w, x):
    return 1/(1+np.exp(-(np.dot(x, w)), dtype=np.float128))


with open("test.csv", "r", encoding='latin1') as test:
    testfile = pd.read_csv(test, header=None)

testfile = testfile.values.tolist()

labeltest = []
smstest = []

# store the label and the sms into two seperated list
for line in testfile:
    labeltest.append(line[0])
    smstest.append(line[1])

labeltestnum = np.array([])

# convert the 'ham' into 1, and 'spam' into 0
for i in labeltest:
    if i == 'ham':
        labeltestnum = np.append(labeltestnum, 1)
    if i == 'spam':
        labeltestnum = np.append(labeltestnum, 0)

transformer = TfidfTransformer(smooth_idf=True)

ct = 0
i = 0

# count the number of right match.
for sms in smstest:

    # apply the same dict to vectorize the test set
    testarray = vectorizer.transform([sms]).toarray()

    # tf_idf transform
    testarray = transformer.fit_transform(testarray).toarray()

    # compute the result of sigmoid function, using w_opt
    result = sigmoid(w_opt, testarray)

    if result >= 0.5:
        result = 1
    else:
        result = 0

    if result == labeltestnum[i]:
        ct = ct + 1

    i = i + 1

print('the number of right match is:', ct)
print('the accuracy rate is:', 100 * ct / len(labeltestnum))

exit()