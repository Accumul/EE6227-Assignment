from scipy.io import loadmat
import numpy as np
import pandas as pd
import math


def import_data():
    data_train = loadmat('data/Data_Train.mat')['Data_Train']  # 120*4
    data_test = loadmat('data/Data_Test.mat')['Data_test']  # 30*4
    label_train = loadmat('data/Label_Train.mat')['Label_Train']  # 120*1
    return data_train, label_train, data_test

data_train, label_train, data_test=import_data()

print(data_train)

def data_partition(verify_size):
    data_verify = []
    label_verify = []
    data_train, label_train, data_test = import_data()
    for i in range(verify_size):
        n = np.random.randint(0, len(data_train))
        data_verify.append(data_train[n])
        label_verify.append(label_train[n])
        data_train = np.delete(data_train, n, 0)
        label_train = np.delete(label_train, n, 0)
    data_verify = np.array(data_verify)
    label_verify = np.array(label_verify)
    return data_train, label_train, data_verify, label_verify, data_test

res=[]
for _ in range(10):
    data_train, label_train, data_verify, label_verify, data_test = data_partition(30)
    pd.set_option('display.max_rows', None)
    df = pd.DataFrame(data_train, index=['row%d' % i for i in range(1, 91)])
    df.columns = ['ft1', 'ft2', 'ft3', 'ft4']
    df['label'] = label_train

    df1 = df[df['label'].isin([1])]
    df2 = df[df['label'].isin([2])]
    df3 = df[df['label'].isin([3])]
    ar1 = np.array(df1)
    ar2 = np.array(df2)
    ar3 = np.array(df3)
    u1 = ar1.mean(axis=0)[0:4]
    u2 = ar2.mean(axis=0)[0:4]
    u3 = ar3.mean(axis=0)[0:4]
    s1 = np.zeros((4, 4))
    s2 = np.zeros((4, 4))
    s3 = np.zeros((4, 4))
    for i in range(len(ar1)):
        diff = np.array([ar1[i, 0:4] - u1])
        s1 += np.dot(diff.T, diff)
    s1 = s1 / len(ar1)
    for i in range(len(ar2)):
        diff = np.array([ar2[i, 0:4] - u2])
        s2 += np.dot(diff.T, diff)
    s2 = s2 / len(ar2)
    for i in range(len(ar3)):
        diff = np.array([ar3[i, 0:4] - u3])
        s3 += np.dot(diff.T, diff)
    s3 = s3 / len(ar3)


    def px(x, u, s):
        return math.exp(-0.5 * ((x - u) @ np.linalg.inv(s) @ (x - u))) / (
                ((2 * math.pi) ** (len(s) / 2)) * (np.linalg.det(s) ** 0.5))


    def pw(x):
        return len(x) / len(data_train)


    right = 0
    for i in range(len(data_verify)):
        l = [pw(ar1) * px(data_verify[i], u1, s1), pw(ar2) * px(data_verify[i], u2, s2),
             pw(ar3) * px(data_verify[i], u3, s3)]
        cla = l.index(max(l)) + 1
        if cla == label_verify[i]:
            right += 1
    acc = right / len(data_verify)
    res.append(acc)
print(res)
