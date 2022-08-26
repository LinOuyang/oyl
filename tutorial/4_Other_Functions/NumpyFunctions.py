import numpy as np
import oyl.utils as u

##创建原始数组，是10,20,5,8的shape
d = np.arange(0, 800, 0.1).reshape(10, 20, 5, 8)
print(d.shape)

##combine_dim合并维度，对指定的维度合并
##本例是对倒数两个维度(5,8)合并
##得到的是10, 20, 40
d1 = u.combine_dim(d, -1, -2)
print(d1.shape)

##与combine相反，decompose是分解
##第一个参数是要分解的维度，本例是1，就是shape里面的20
##后面的参数是要分解成的形状，分解成2和一个-1
d2 = u.decompose_dim(d, 1, [2, -1])
print(d2.shape)


print("="*20)
"""
使用utils里面的stat函数可以快速print一个np的数组的信息
依次为：缺失数据的占比，数组shape，均方差
最小值、平均值、最大值
"""
u.stat(d, name='OrginData')


##使用utils里面的view函数可以快速imshow一个二维数组
u.view(d[..., 0, 0], cmap='rainbow')
