# -*- coding = utf-8 -*-
# @Time : 2022/3/28 13:39
# @Author : CDC
# @File : Data_visu.py
# @Software: PyCharm
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data_train = loadmat('data/Data_Train.mat')['Data_Train']  # 120*4
data_test = loadmat('data/Data_Test.mat')['Data_test']  # 30*4
label_train = loadmat('data/Label_Train.mat')['Label_Train']  # 120*1
data_train = data_train[data_train[:,3].argsort()]
print(data_train[40:84])
x, y, z = data_train[:,0], data_train[:,1], data_train[:,2]

# ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程

fig = plt.figure()
ax = Axes3D(fig)
#  将数据点分成三部分画，在颜色上有区分度
ax.scatter(x[0:40], y[0:40], z[0:40], c='y')  # 绘制数据点
ax.scatter(x[40:84], y[40:84], z[40:84], c='r')
ax.scatter(x[84:120], y[84:120], z[84:120], c='g')
ax.view_init(elev=10,    # 仰角
             azim=300    # 方位角
            )

ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
ax.set_title('title 1')
plt.savefig('./test2.png')

plt.show()
