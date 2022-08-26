import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from oyl.nn import RBFRegressor


def score(m, X, y):
    """
    r2_score是衡量回归模型好坏的一个指标，区间在(-inf, 1)
    越接近1代表回归得越好
    """
    pred = m.predict(X).flatten()
    true = y.flatten()
    return r2_score(true, pred)

##读取数据
d = pd.read_csv("../Datas/boston_data.csv",index_col=0).to_numpy()

##划分X和Y
X, y = d[:,:-1], d[:, -1:]
##前350个训练，剩下的验证
XTrain = X[:350]
XVal = X[350:]
YTrain = y[:350]
YVal = y[350:]

##建立隐层为500个神经元的模型
m = MLPRegressor(100, max_iter=1000)
##训练
m.fit(XTrain, YTrain.squeeze())
##训练集的得分
print(f"MLP TrainSet score : {score(m, XTrain, YTrain):.2f}")
##验证集的得分
print(f"MLP ValSet score : {score(m, XVal, YVal):.2f}")

##建立隐层为100个神经元的径向基函数神经网络模型
m = RBFRegressor(100,  0.1)
##训练
m.fit(XTrain, YTrain.squeeze())
##训练集的得分
print(f"RBFN TrainSet score : {score(m, XTrain, YTrain):.2f}")
##验证集的得分
print(f"RBFN ValSet score : {score(m, XVal, YVal):.2f}")

