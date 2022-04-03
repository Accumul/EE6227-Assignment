# -*- coding = utf-8 -*-
# @Time : 2022/3/25 2:18
# @Author : CDC
# @File : Decision Tree.py
# @Software: PyCharm
from scipy.io import loadmat
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pydotplus
from sklearn import tree


def import_data():
    data_train = loadmat('data/Data_Train.mat')['Data_Train']  # 120*4
    data_test = loadmat('data/Data_Test.mat')['Data_test']  # 30*4
    label_train = loadmat('data/Label_Train.mat')['Label_Train']  # 120*1
    return data_train, label_train.ravel(), data_test


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
    return data_train, label_train.ravel(), data_verify, label_verify, data_test


data_train, label_train, data_verify, label_verify, data_test = data_partition(30)

clf = DecisionTreeClassifier(max_depth=3)
clf.fit(data_train, label_train)
dot_data = tree.export_graphviz(clf, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_pdf("tree structure.pdf")
op = clf.predict(data_verify)

# Prediction
right = 0
for i in range(len(data_verify)):
    if op[i] == label_verify[i]:
        right += 1
acc = right / len(data_verify)
print(acc)



# def tree(data, label):
#     ziplist = list(zip(data, label))
#     ziplist.sort()
#     split_list = []
#     imp_list = []
#     for i in range(1, len(ziplist)):
#         left = ziplist[:i]
#         right = ziplist[i:]
#         l1, l2 = zip(*left)
#         r1, r2 = zip(*right)
#         lcl1 = l2.count(1)
#         lcl2 = l2.count(2)
#         lcl3 = l2.count(3)
#         rcl1 = r2.count(1)
#         rcl2 = r2.count(2)
#         rcl3 = r2.count(3)
#         ginileft = (1 - (lcl1 / len(left)) ** 2 - (lcl2 / len(left)) ** 2 - (lcl3 / len(left)) ** 2) / 2
#         giniright = (1 - (rcl1 / len(right)) ** 2 - (rcl2 / len(right)) ** 2 - (rcl3 / len(right)) ** 2) / 2
#         imp = len(left) / len(ziplist) * ginileft + len(right) / len(ziplist) * giniright
#         split_list.append((ziplist[i-1][0] + ziplist[i][0]) / 2)
#         imp_list.append(imp)
#     miniimp = min(imp_list)
#     splitpoint = imp_list.index(miniimp)  # 在下标为m的点之后分割
#     splitvalue = (ziplist[splitpoint][0] + ziplist[splitpoint + 1][0]) / 2
#     plt.plot(split_list, imp_list,marker = 'o', markersize=5,mec = 'r', mfc = 'w')
#     plt.grid(True)
#     plt.xlabel('Split value')
#     plt.ylabel('Impurity')
#     plt.title('Impurity trend')
#     plt.savefig('./test2.png')
#     plt.show()
#
#     return miniimp, splitvalue
#
# fourDparaImp=[]
# fourDparaSV=[]
#
# for i in range(1):
#     imp, splitvalue = tree(data_train[:, i], label_train)
#     fourDparaImp.append(imp)
#     fourDparaSV.append(splitvalue)
# print(fourDparaImp)
# print(fourDparaSV)
