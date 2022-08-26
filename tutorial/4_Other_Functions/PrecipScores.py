import numpy as  np
import oyl.utils as u

##生成随机的观测降雨数据
obs = np.abs(np.random.normal(0, 30, [3404, 66, 81]))
##对预报的降雨数据生成随机偏差（偏大或偏小
pre = obs * np.random.normal(0.9, 0.2, obs.shape)

"""
对于一般的降水预报问题，ScoreFuns收录了许多评分函数如下：
acc : 准确率
bias : 偏差（预报得比实际的是更旱或更湿
ts :  风险评分
ets : 公平技巧评分
hss : hss评分
pod : 命中率
far : 空报率
"""

"""
输入预测和实际的降水，建立一个object
axis指定对时间方向做评分（每个格点都存在一个评分）
当只计算一维数组的得分时可以不考虑该参数
threshold是降水阈值，它能够衡量该量级降水的得分
object建立后，使用评分函数的名字(如self.ts() 可以得到计算的得分
当需要计算多个指标时，可以使用self.get方法传入函数名，不区分大小写
self.get返回的是列表
"""

score = u.ScoreFuns(obs, pre, threshold=5, axis=0)
far = score.far()
ets, = score.get('ETS')
acc, ts = score.get(['Acc', 'Ts'])

##查看得分
u.view(acc)

