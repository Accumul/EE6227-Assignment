from scipy.io import loadmat
import numpy as np
import pandas as pd


def import_data():
    data_train = loadmat('data/Data_Train.mat')['Data_Train']  # 120*4
    data_test = loadmat('data/Data_Test.mat')['Data_test']  # 30*4
    label_train = loadmat('data/Label_Train.mat')['Label_Train']  # 120*1
    return data_train, label_train, data_test


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


res = []
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
    m1 = ar1.mean(axis=0)[0:4]
    m2 = ar2.mean(axis=0)[0:4]
    m3 = ar3.mean(axis=0)[0:4]
    m = (m1 * len(ar1) + m2 * len(ar2) + m3 * len(ar3)) / (len(data_train))
    s1 = np.zeros((4, 4))
    s2 = np.zeros((4, 4))
    s3 = np.zeros((4, 4))
    for i in range(len(ar1)):
        diff = np.array([ar1[i, 0:4] - m1])
        s1 += np.dot(diff.T, diff)
    for i in range(len(ar2)):
        diff = np.array([ar2[i, 0:4] - m2])
        s2 += np.dot(diff.T, diff)
    for i in range(len(ar3)):
        diff = np.array([ar3[i, 0:4] - m3])
        s3 += np.dot(diff.T, diff)
    sw = (len(ar1) * s1 + len(ar1) * s2 + len(ar1) * s3) / len(data_train)
    # print(sw)
    st = np.zeros((4, 4))
    for i in range(len(data_train)):
        st += np.dot(np.array([data_train[i] - m]).T, np.array([data_train[i] - m]))
    # print(st)
    # sb1 = len(ar1) * np.dot(np.array([m1 - m]).T, np.array([m1 - m]))
    # sb2 = len(ar2) * np.dot(np.array([m2 - m]).T, np.array([m2 - m]))
    # sb3 = len(ar3) * np.dot(np.array([m3 - m]).T, np.array([m3 - m]))
    # sb = sb1 + sb2 + sb3
    sb = st - sw
    # print(sb)

    class_num = 3
    evalue, evector = np.linalg.eig(np.linalg.inv(sw) @ sb)

    w1 = evector[:, list(evalue).index(max(evalue))].real
    # w2 = evector[:, 1].real
    # print(evalue)
    # print(evector)
    m11hat = 0
    m12hat = 0
    m13hat = 0
    for i in range(len(ar1)):
        m11hat += np.dot(w1, ar1[i, 0:4])
    m11hat /= len(ar1)
    for i in range(len(ar2)):
        m12hat += np.dot(w1, ar2[i, 0:4])
    m12hat /= len(ar2)
    for i in range(len(ar3)):
        m13hat += np.dot(w1, ar3[i, 0:4])
    m13hat /= len(ar3)
    m_hat = [m11hat, m12hat, m13hat]
    m_hat_sort = sorted(m_hat)
    cla1 = m_hat.index(m_hat_sort[0]) + 1
    cla2 = m_hat.index(m_hat_sort[1]) + 1
    cla3 = m_hat.index(m_hat_sort[2]) + 1
    # print(cla1, cla2, cla3)
    w01 = -(m_hat_sort[0] + m_hat_sort[1]) / (class_num - 1)
    w02 = -(m_hat_sort[1] + m_hat_sort[2]) / (class_num - 1)


    # m21hat = 0
    # m22hat = 0
    # m23hat = 0
    # for i in range(len(ar1)):
    #     m21hat += np.dot(w2, ar1[i, 0:4])
    # m21hat /= len(ar1)
    # for i in range(len(ar2)):
    #     m22hat += np.dot(w2, ar2[i, 0:4])
    # m22hat /= len(ar2)
    # for i in range(len(ar3)):
    #     m23hat += np.dot(w2, ar3[i, 0:4])
    # m23hat /= len(ar3)
    # # class3 is seperated, class1 is closer
    # w02 = -(m21hat + m23hat) / (class_num - 1)

    # print(m_hat)
    # print(m_hat_sort)

    # print(m21hat, m22hat, m23hat)

    def gx(x, w, w0):
        return np.dot(w, x) + w0


    right = 0
    for i in range(len(data_verify)):
        if gx(data_verify[i], w1, w02) > 0:
            cla = m_hat.index(m_hat_sort[2]) + 1
        elif gx(data_verify[i], w1, w01) < 0:
            cla = m_hat.index(m_hat_sort[0]) + 1
        else:
            cla = m_hat.index(m_hat_sort[1]) + 1

        if cla == label_verify[i]:
            right += 1
    acc = right / len(data_verify)
    res.append(acc)
print(res)
