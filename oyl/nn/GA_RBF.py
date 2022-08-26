import numpy as np
from random import randint
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.cluster import KMeans
from time import sleep


class GA_RBFRegressor:
    def __init__(self, pop=5, max_iter=200, hidden_layer_size=10,  times=0.5, fit_intercept=False):
        self.hidden_layer_size = hidden_layer_size
        self.times = times
        self.pop = pop
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept

    def _fit(self, x, y):
        num = len(x)
        if num>1000:
            rest_time = 10
        else:
            rest_time = 0
        print(rest_time)
        std_x, self.scale_x = self.standard(x)
        std_y, self.scale_y = self.standard(y)
        if len(y.shape) == 1:
            self.output_layer_size = 1
        else:
            self.output_layer_size = y.shape[1]
        # 确定隐层的核中心和基宽
        km = KMeans(n_clusters=self.hidden_layer_size,n_jobs=3)
        km.fit(std_x)
        self.center = [i for i in km.cluster_centers_]
        self.width = [np.mean(np.sqrt(np.sum(np.power(std_x-i, 2), axis=1))) for i in km.cluster_centers_]
        # 用遗传算法确定隐层到输出层的权重和截距（偏置）
        # 初始化种群
        population_w, population_b = list(), list()
        for j in range(self.pop):
            one_w = np.random.uniform(-1, 1, [self.hidden_layer_size, self.output_layer_size])
            population_w.append(one_w)
            one_b = np.random.uniform(-1, 1, self.output_layer_size)
            population_b.append(one_b)
        # 开始遗传优化
        best_pop = [np.inf,0,0]
        iteration = 0
        while iteration <= self.max_iter:
            new_pop_w, new_pop_b=list(),list()
            # 计算适应度并记录最好个体
            pop_fitness = [self.fitness(std_x, std_y, population_w[i], population_b[i]) for i in range(len(population_w))]
            if min(pop_fitness) < best_pop[0]:
                n = pop_fitness.index(min(pop_fitness))
                best_pop = [pop_fitness[n], population_w[n], population_b[n]]
            for i in range(int(self.pop/2)):
                # 选择操作
                x, y = self.select(pop_fitness)
                # 将个体建设为染色体
                person1_w, person2_w = population_w[x].reshape(-1), population_w[y].reshape(-1)
                # 交叉互换
                person1_w, person2_w = self.cross(person1_w, person2_w) 
                # 变异
                person1_w = self.mut(person1_w)
                person2_w = self.mut(person2_w)
                # 将染色体进化为个体
                person1_w.shape = self.hidden_layer_size, self.output_layer_size
                person2_w.shape = self.hidden_layer_size, self.output_layer_size
                # 构建新的种群
                new_pop_w += [person1_w,person2_w]

                if self.fit_intercept:
                    person1_b, person2_b = self.matrix_to_vector(population_b[x]), self.matrix_to_vector(population_b[y])
                    person1_b, person2_b = self.cross(person1_b, person2_b)
                    person1_b = self.mut(person1_b)
                    person2_b = self.mut(person2_b)
                    person1_b = np.array(person1_b)
                    person2_b = np.array(person2_b)
                    new_pop_b += [person1_b,person2_b]
                else:
                    new_pop_b += [0, 0]
                
            new_pop_w.append(best_pop[1])
            new_pop_b.append(best_pop[2])
            population_w,population_b=new_pop_w,new_pop_b
            iteration += 1
            if iteration%20==0 and rest_time!=0:
                print('({}/{})resting'.format(iteration,self.max_iter))
                sleep(rest_time)     
        else:
            # 遗传进化完成，计算适应度并记录最好个体
            pop_fitness = [self.fitness(std_x, std_y, population_w[i], population_b[i]) for i in
                           range(len(population_w))]
            if min(pop_fitness) < best_pop[0]:
                n = pop_fitness.index(min(pop_fitness))
                best_pop = [pop_fitness[n], population_w[n], population_b[n]]
        self.weight, self.intercepts = best_pop[1], best_pop[2]


    def fit(self, x, y):
        self._fit(x, y)

    def _predict(self, x):
        std_x = self.standard(x, scale=self.scale_x )
        hidden_outputs = [
            np.exp(-np.sqrt(np.sum(np.power(std_x - self.center[i], 2), axis=1)) / (np.power(self.width[i], 2))) for
            i in range(self.hidden_layer_size)]
        hidden_outputs = np.array(hidden_outputs).T
        if not self.fit_intercept:
            self.intercepts = 0
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

    # 遗传算法的一些操作所需函数
    def fitness(self,x,y,w,b):
        hidden_outputs = [
            np.exp(-np.sqrt(np.sum(np.power(x - self.center[i], 2), axis=1)) / (np.power(self.width[i], 2))) for
            i in range(self.hidden_layer_size)]
        hidden_outputs = np.array(hidden_outputs).T
        if not self.fit_intercept:
            b = 0
        pre = np.dot(hidden_outputs, w)+b
        if len(y.shape)==1:
            pre.shape = y.shape
        dist =np.sqrt(np.sum(np.power(pre-y,2)))
        return dist
    
    def select(self,f,d=15):
        # 根据适应度选择两个个体，d为选择程度，适应度越小越容易被选中
        n=len(f)
        f=np.array(f)
        p=pow(1/f,d)
        p=p/sum(p)
        # 以上求倒数乘方并归一化得出选取概率，以下产生随机数选取
        ra=np.random.random()
        s=0  # 累积概率
        for i in range(n):
            s+=p[i]
            if ra<=s:
                x=i
                break
        p[x]=0
        p=p/sum(p)
        ra=np.random.random()
        s=0
        for i in range(n):
            s+=p[i]
            if ra<=s:
                y=i
                break
        return x,y

    def cross(self, x, y):
        n = len(x)
        a, b = randint(0,n-1), randint(0,n-1)
        [a,b] = np.sort([a,b])
        (x[a:b+1], y[a:b+1]) = (y[a:b+1], x[a:b+1])
        return x, y

    def mut(self,x):
        n=len(x)
        a,b=randint(0,n-1),randint(0,n-1)
        [a,b]=np.sort([a,b])
        random_value = np.random.uniform(-1,1,[1,b-a+1])
        x[a:b+1] = np.array(x[a:b+1]) + random_value
        return x






class GA_RBFClassifier(GA_RBFRegressor):
    def __init__(self,pop=5, max_iter=200, hidden_layer_size=10, times=0.5, fit_intercept=False):
        super().__init__(pop=pop, max_iter=max_iter, hidden_layer_size=hidden_layer_size, times=times, fit_intercept=fit_intercept)

    def fit(self, x, y):
        cate = list()
        num_cate = 0
        for i in y:
            if i not in cate:
                num_cate += 1
                cate.append(i)
        y_dist = -1000*np.ones([len(y), num_cate])
        for i in range(len(y)):
            y_dist[i, cate.index(y[i])] = 1000
        self.classes_ = np.array(cate)
        self._fit(x, y_dist)

    def predict_proba(self,x):
        raw_pro = self._predict(x)
        for i in range(len(raw_pro)):
            raw_pro[i, :] = raw_pro[i, :]-np.min(raw_pro[i, :])
            s = np.sum(raw_pro[i, :])
            raw_pro[i, :] = raw_pro[i, :]/s
        return raw_pro

    def predict(self, x):
        pro = self._predict(x)
        most_pro_index = np.argmax(pro, axis=1)
        cate = list()
        for i in most_pro_index:
            cate.append(self.classes_[i])
        pre = np.array(cate)
        if 1 in pre.shape:
            pre.shape=-1
        return pre

    def score(self,x,y):
        pre_y = self.predict(x)
        right=0
        n=len(pre_y)
        for i in range(n):
            if pre_y[i] == y[i]:
                right += 1
        return right/n



if __name__=='__main__':
    view = '分类回归分析'
    if view == '数值回归分析':
        # 数值回归分析
        s1 = GA_RBFRegressor(hidden_layer_size=3, pop=8, max_iter=1000, times=0.6, fit_intercept=True)
        s2 = MLPRegressor(hidden_layer_sizes=10, max_iter=800, activation='relu')
        # y=x1+2lnx2-3x3
        x = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15],[16,17,18]])
        y = x[:, 0]+2*np.log(x[:, 1])-3*x[:, 2]
        y.shape = -1, 1
        test_x = np.array([[19,20,21],[22,23,24],[25,26,27]])
        test_y = test_x[:,0]+2*np.log(test_x[:,1])-3*test_x[:,2]
        test_y.shape = -1, 1
        s1.fit(x, y)
        s2.fit(x, y)
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
        s1 = GA_RBFClassifier(hidden_layer_size=3,pop=8, max_iter=500, times=0.6, fit_intercept=False)
        s2 = MLPClassifier(hidden_layer_sizes=[10,2],max_iter=800,activation='relu')
        x = np.array([[10, 2, 3], [4, 15, 6], [7, 8, 11], [1, 9, 12], [5, 14, 5], [16, 7, 8]])
        y = np.array(['a','b','c','c','b','a'])
        y.shape = -1, 1
        #x第一列超过10就是a,2列超过是b，3列超过是c
        test_x = np.array([[1, 14, 2], [12, 3, 4], [5, 6, 17],[3,11,7]])
        test_y = np.array(['b','a','c','b'])
        test_y.shape = -1, 1
        s1.fit(x, y)
        s2.fit(x, y)
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
