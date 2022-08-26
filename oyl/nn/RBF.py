import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder

class RBFRegressor:
    def __init__(self, hidden_layer_sizes=10,  times=0.5, fit_intercept=False):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.times = times
        self.fit_intercept = fit_intercept

    def _fit(self, x, y):
        std_x, self.scale_x = self.standard(x)
        std_y, self.scale_y = self.standard(y)
        if len(y.shape) == 1:
            output_layer_sizes = 1
        else:
            output_layer_sizes = y.shape[1]
        km = KMeans(n_clusters=self.hidden_layer_sizes)
        km.fit(std_x)
        hidden_nodes_c = [i for i in km.cluster_centers_]
        hidden_nodes_w = [np.mean(np.sqrt(np.sum(np.power(std_x-i, 2), axis=1))) for i in km.cluster_centers_]
        hidden_outputs = [np.exp(-np.sqrt(np.sum(np.power(std_x-hidden_nodes_c[i], 2), axis=1))/(np.power(hidden_nodes_w[i],2))) for i in range(self.hidden_layer_sizes)]
        hidden_outputs = np.array(hidden_outputs).T
        weights, intercepts = list(), [0 for i in range(output_layer_sizes)]
        for i in range(output_layer_sizes):
            if y.ndim==1:
                y_for_fit = std_y
            else:
                y_for_fit = std_y[:,i]
            lr = LinearRegression(fit_intercept=self.fit_intercept)
            lr.fit(hidden_outputs,y_for_fit)
            w = lr.coef_
            weights.append(w)
            if self.fit_intercept:
                b = lr.intercept_
                intercepts[i] = b
        weights = np.array(weights).T
        self.center = hidden_nodes_c
        self.width = hidden_nodes_w
        self.weight = weights
        self.intercepts = np.array(intercepts)

    def fit(self, x, y):
        self._fit(x, y)

    def _predict(self, x):
        std_x = self.standard(x, scale=self.scale_x )
        hidden_outputs = [
            np.exp(-np.sqrt(np.sum(np.power(std_x - self.center[i], 2), axis=1)) / (np.power(self.width[i], 2))) for
            i in range(self.hidden_layer_sizes)]
        hidden_outputs = np.array(hidden_outputs).T
        output = np.dot(hidden_outputs,self.weight)+self.intercepts
        pre = self.unstandard(output, self.scale_y)
        return pre

    def predict(self, x):
         pre = self._predict(x)
         if 1 in pre.shape:
             pre.shape = -1
         return pre


    def score(self, x, y):
        pred_y = self.predict(x)
        if len(y.shape) == 1:
            pred_y.shape = y.shape
        u = np.sum(np.power(pred_y-y,2))
        v = np.sum(np.power(pred_y-np.mean(pred_y, axis=0), 2))
        return 1-u/v

    def standard(self, data, scale=0):
        #数据标准化，返回标准化后的数据和数据的规格，该规格是2行的n列的矩阵列表，第一行表示减去的值，第二行表示长度
        if scale==0:
            a = np.min(data, axis=0)
            b = np.max(data, axis=0)-a
            a = a - self.times*b
            b = (2*self.times+1)*b
            std_data=(data-a)/b
            scale_data=[a,b]
            return std_data,scale_data
        else:
            std_data=(data-scale[0])/scale[1]
            return std_data

    def unstandard(self,std_data,scale_data):
        a=scale_data[0]
        b=scale_data[1]
        data = std_data*b+a
        return data



class RBFClassifier(RBFRegressor):
    def __init__(self, hidden_layer_sizes=10, times=0.3, fit_intercept=False):
        super().__init__(hidden_layer_sizes=hidden_layer_sizes, times=times, fit_intercept=fit_intercept)

    def fit(self, x, y):
        y = np.array(y).reshape(-1,1)
        self.ohe = OneHotEncoder(sparse=False)
        y_dist = self.ohe.fit_transform(y.reshape(-1,1))
        self._fit(x, y_dist)

    def predict_proba(self,x):
        raw_pro = self._predict(x)
        raw_pro = raw_pro - np.min(raw_pro,axis=1).reshape(-1,1)
        s = np.sum(raw_pro,axis=1)
        pro = raw_pro/s.reshape(-1,1)
        return pro

    def predict(self, x):
        pro = self._predict(x)        
        pre = self.ohe.inverse_transform(pro)
        return pre.reshape(-1)

    def score(self,x,y):
        pre_y = self.predict(x)
        n=len(pre_y)
        right = (pre_y==y).sum()
        return right/n







if __name__=='__main__':
    view = '分类回归分析'
    if view == '数值回归分析':
        # 数值回归分析
        s1 = RBFRegressor(hidden_layer_sizes=3,times=1,fit_intercept=True)
        s2 = MLPRegressor(hidden_layer_sizes=10, max_iter=800, activation='relu')
        # y=x1+2lnx2-3x3
        x = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15],[16,17,18]])
        y = x[:, 0]+2*np.log(x[:, 1])-3*x[:, 2]
        y.shape = -1, 1
        test_x = np.array([[19,20,21],[22,23,24],[25,26,27]])
        test_y = test_x[:,0]+2*np.log(test_x[:,1])-3*test_x[:,2]
        test_y.shape = -1, 1
        s1.fit(x, y)
        s2.fit(x, y.reshape(-1))
        pre1_y = s1.predict(test_x)
        pre2_y = s2.predict(test_x)
        print('实际值：')
        print(test_y)
        print(' ')
        print('RBF网络输出和得分：')
        print(pre1_y)
        print(s1.score(test_x,test_y))
        print(' ')
        print('多层感知器输出和得分：')
        print(pre2_y)
        print(s2.score(test_x,test_y))
    elif view == '分类回归分析':
        # 分类回归分析
        s1 = RBFClassifier(hidden_layer_sizes=3,times=0.5)
        s2 = MLPClassifier(hidden_layer_sizes=10,max_iter=800,activation='relu')
        x = np.array([[10, 2, 3], [4, 15, 6], [-1.7, 8, 11], [-1.2, 9, 12], [5, 14, 5], [16, 7, 8]])
        y = np.array(['a','b','c','c','b','a'])
        y.shape = -1, 1
        #x第一列超过10就是a,2列超过是b，3列超过是c
        test_x = np.array([[1, 14, 2], [12, 3, 4], [5, 6, 17],[3,11,7]])
        test_y = np.array(['b','a','c','b'])
        s1.fit(x, y)
        s2.fit(x, y.reshape(-1))
        pre1_y = s1.predict(test_x)
        pre2_y = s2.predict(test_x)
        pro1 = s1.predict_proba(test_x)
        pro2 = s2.predict_proba(test_x)
        print('RBF网络输出概率和分类以及准确率：')
        print(pro1)
        print(pre1_y)
        print(s1.score(test_x, test_y))
        print('实际分类')
        print(test_y)
        print('多层感知器输出概率和分类以及准确率：')
        print(pro2)
        print(pre2_y)
        print(s2.score(test_x, test_y))
