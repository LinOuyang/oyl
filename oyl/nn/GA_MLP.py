import numpy as np
from random import randint
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import OneHotEncoder
import warnings

class GA_MLPRegressor:

    def __init__(self, hidden_layer_sizes=[10], activation='relu', times=1, pop=5, max_iter=500):
        if type(hidden_layer_sizes) == list:
            self.hidden_layer_sizes = hidden_layer_sizes
        elif type(hidden_layer_sizes) == int:
            self.hidden_layer_sizes = [hidden_layer_sizes]
        self.activation = activation
        self.pop = pop
        self.max_iter = max_iter
        self.times = times
        if activation == 'logistic':
            self.method = '0_1'
        elif activation == 'relu':
            self.method = '0_inf'

    def _fit(self, x, y, class_weights=[1]):
        if (len(class_weights)>1) and (len(class_weights)<y.shape[1]):
            raise TypeError('参数class_weights长度与y的类别不匹配')
        try:
            std_x, self.scale_x = self.standard(x, method = self.method)
            std_y, self.scale_y = self.standard(y ,method = self.method)
            self.layers = [std_x.shape[1]] + self.hidden_layer_sizes + [std_y.shape[1] if std_y.ndim ==2 else 1]
            # 初始化种群
            population_w,population_b = list(),list()
            for j in range(self.pop):
                one_w,one_b=list(),list()      
                for i in range(len(self.layers)-1):
                    one_w.append(np.random.uniform(-1, 1, [self.layers[i], self.layers[i+1]]))
                    one_b.append(np.random.uniform(-1, 1, [1,self.layers[i+1] ]))
                population_w.append(one_w)
                population_b.append(one_b)
            # 开始遗传优化
            best_pop = [np.inf,0,0]
            iteration = 0
            while iteration <= self.max_iter:
                new_pop_w,new_pop_b=list(),list()
                # 计算适应度并记录最好个体
                pop_fitness=list()
                for i in range(len(population_w)):
                    pop_fitness.append(self.fitness(std_x,std_y,population_w[i],population_b[i],class_weights=class_weights))
                if min(pop_fitness) < best_pop[0]:
                    n=pop_fitness.index(min(pop_fitness))
                    best_pop = [pop_fitness[n] , population_w[n] , population_b[n]]
                for i in range(int(self.pop/2)):
                    # 选择操作
                    x,y=self.select(pop_fitness)
                    # 将个体建设为染色体
                    person1_w,person2_w = self.matrix_to_vector(population_w[x]), self.matrix_to_vector(population_w[y])
                    person1_b,person2_b = self.matrix_to_vector(population_b[x]), self.matrix_to_vector(population_b[y])
                    # 交叉互换
                    person1_w, person2_w, person1_b, person2_b = self.cross(person1_w, person2_w, person1_b, person2_b)
                    # 变异
                    person1_w=self.mut(person1_w)
                    person2_w=self.mut(person2_w)
                    person1_b=self.mut(person1_b)
                    person2_b=self.mut(person2_b)
                    # 将染色体进化为个体
                    person1_w = self.vector_to_weights(person1_w)
                    person2_w = self.vector_to_weights(person2_w)
                    person1_b = self.vector_to_bias(person1_b)
                    person2_b = self.vector_to_bias(person2_b)
                    # 构建新的种群
                    new_pop_w += [person1_w,person2_w]
                    new_pop_b += [person1_b,person2_b]
                new_pop_w.append(best_pop[1])
                new_pop_b.append(best_pop[2])
                population_w,population_b=new_pop_w,new_pop_b
                iteration += 1
                if iteration%100==0:
                    print('({}/{})complete'.format(iteration,self.max_iter))
                
            else:
                # 遗传进化完成，计算适应度并记录最好个体
                pop_fitness = [self.fitness(std_x, std_y, population_w[i], population_b[i],class_weights=class_weights) for i in
                               range(len(population_w))]
                if min(pop_fitness) < best_pop[0]:
                    n = pop_fitness.index(min(pop_fitness))
                    best_pop = [pop_fitness[n], population_w[n], population_b[n]]
                self.w, self.b = best_pop[1], best_pop[2]
        except KeyboardInterrupt:
            pop_fitness = [self.fitness(std_x, std_y, population_w[i], population_b[i]) for i in
                           range(len(population_w))]
            if min(pop_fitness) < best_pop[0]:
                n = pop_fitness.index(min(pop_fitness))
                best_pop = [pop_fitness[n], population_w[n], population_b[n]]
            self.w, self.b = best_pop[1], best_pop[2]
            warnings.warn('Training interrupted by user.')


    def fit(self, x, y):
        self._fit(x,y)

    def _predict(self, x):
        std_x = self.standard(x, method = self.method, scale = self.scale_x )
        out_puts=self.activate(np.dot(std_x, self.w[0]) - self.b[0])
        for i in range(len(self.layers)-2):
            out_puts=self.activate(np.dot(out_puts, self.w[i+1]) - self.b[i+1])
        pre = self.unstandard(out_puts, self.scale_y)
        if 1 in pre.shape:
            pre.shape = -1
        return pre

    def predict(self,x):
        return self._predict(x)


    def activate(self,x):
        if self.activation == 'logistic':
            return  (1 / (1 + np.exp(-x)))
        elif self.activation == 'relu':
            y=x.copy()
            y[y<0]=0
            return y

    def score(self,x,y):
        pred_y = self.predict(x)
        if len(y.shape)==1:
            pred_y.shape = y.shape
        u = np.sum(np.power(pred_y-y,2))
        v = np.sum(np.power(pred_y-np.mean(pred_y,axis=0),2))
        return 1-u/v

    #数据处理的一些操作所需函数
        
    def standard(self, data, method='0_1', scale=0):
        #数据标准化，返回标准化后的数据和数据的规格，该规格是2行的n列的矩阵列表，第一行表示减去的值，第二行表示长度 
        if scale==0:
            a = np.min(data, axis=0)
            b = np.max(data, axis=0)-a
            a = a - self.times*b
            b = (2*self.times+1)*b
            if method == '0_1':
                pass
            elif method == '0_inf':
                b=1
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
        
    def matrix_to_vector(self,one):
        vec=list()
        for i in range(len(one)):
            vec+=(one[i].reshape(1,-1).tolist()[0])
        return vec

    def vector_to_weights(self,vec):
        mat=list()
        v=vec.copy()
        for i in range(len(self.layers)-1):
            num=self.layers[i]*self.layers[i+1]
            tem=np.array(v[0:num])
            tem.shape=self.layers[i],self.layers[i+1]
            del v[0:num]
            mat.append(tem)
        return mat

    def vector_to_bias(self,vec):
        bia=list()
        v=vec.copy()
        for i in range(len(self.layers)-1):
            bia.append(np.array(v[0:self.layers[i+1]]))
            del v[0:self.layers[i+1]]
        return bia
        
        

    #遗传算法的一些操作所需函数

    def fitness(self,x,y,w,b,class_weights):
        out_puts = self.activate(np.dot(x, w[0]) - b[0])
        for i in range(len(self.layers) - 2):
            out_puts = self.activate(np.dot(out_puts, w[i + 1]) - b[i + 1])
        pre = out_puts
        if len(y.shape)==1:
            pre.shape = y.shape
        ans = np.abs(pre-y)*np.array(class_weights)
        dist =np.sqrt(np.sum(np.power(ans,2)))
        return dist
    
    def select(self,f,d=15):
        #根据适应度选择两个个体，d为选择程度，适应度越小越容易被选中
        n=len(f)
        f=np.array(f)
        p=pow(1/f,d)
        p=p/sum(p)
        #以上求倒数乘方并归一化得出选取概率，以下产生随机数选取
        ra=np.random.random()
        s=0  #累积概率
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

    def cross(self,x_1,y_1,x_2,y_2):
        n=len(x_1)
        #随机产生交叉位
        a,b=randint(0,n-1),randint(0,n-1)
        [a,b]=np.sort([a,b])
        (x_1[a:b+1],y_1[a:b+1])=(y_1[a:b+1],x_1[a:b+1])
        (x_2[a:b + 1], y_2[a:b + 1]) = (y_2[a:b + 1], x_2[a:b + 1])
        return x_1,y_1,x_2,y_2

    def mut(self,x):
        n=len(x)
        #随机产生变异位
        a,b=randint(0,n-1),randint(0,n-1)
        [a,b]=np.sort([a,b])
        random_value = np.random.uniform(-1,1,[1,b-a+1])
        x[a:b+1] = (np.array(x[a:b+1]) + random_value).tolist()[0]
        return x


class GA_MLPClassifier(GA_MLPRegressor):
    def __init__(self, hidden_layer_sizes=[10], activation='logistic', times=1, pop=5, max_iter=500):
        super().__init__(hidden_layer_sizes=hidden_layer_sizes, activation=activation, times=times, pop=pop, max_iter=max_iter)

    def fit(self, x, y, class_weights=[1]):
        y = np.array(y).reshape(-1,1)
        self.ohe = OneHotEncoder(sparse=False)
        y_dist = self.ohe.fit_transform(y)        
        self._fit(x, y_dist,class_weights=class_weights)

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
        right=0
        n=len(pre_y)
        for i in range(n):
            if pre_y[i] == y[i]:
                right += 1
        return right/n






    
if __name__=='__main__':
    view = '分类回归分析'
    if view == '数值回归分析':
        #数值回归分析
        s1=GA_MLPRegressor(hidden_layer_sizes=10,max_iter=800,times=1.5,activation='relu')
        s2=MLPRegressor(hidden_layer_sizes=8,max_iter=800,activation='relu')
        #y=x1+2lnx2-3x3
        x=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15],[16,17,18]])
        y=x[:,0]+2*np.log(x[:,1])-3*x[:,2]
        y.shape=-1,1
        test_x=np.array([[19,20,21],[22,23,24],[25,26,27]])
        test_y=test_x[:,0]+2*np.log(test_x[:,1])-3*test_x[:,2]
        test_y.shape=-1,1
        s1.fit(x,y)
        s2.fit(x,y)
        pre1_y=s1.predict(test_x)
        pre2_y=s2.predict(test_x)
        print('实际值：')
        print(test_y)
        print(' ')
        print('遗传改进网络输出和得分：')
        print(pre1_y)
        print(s1.score(test_x,test_y))
        print(' ')
        print('多层感知器输出和得分：')
        print(pre2_y)
        print(s2.score(test_x,test_y))
    elif  view == '分类回归分析':
        #分类回归分析
        s1 = GA_MLPClassifier(hidden_layer_sizes=8,max_iter=200,times=0.05,activation='logistic')
        s2 = MLPClassifier(hidden_layer_sizes=8,max_iter=400,activation='relu')
        x = np.array([[10, 2, 3], [4, 15, 6], [7, 8, 11], [1, 9, 12], [5, 14, 5], [16, 7, 8]])
        y = np.array(['a','b','c','c','b','a'])
        y.shape = -1, 1
        #x第一列超过10就是a,2列超过是b，3列超过是c
        test_x = np.array([[1, 14, 2], [12, 3, 4], [5, 6, 17]])
        test_y = np.array(['b','a','c'])
        test_y.shape = -1, 1
        s1.fit(x, y,[1,9,9])
        s2.fit(x, y)
        pre1_y = s1.predict(test_x)
        pre2_y = s2.predict(test_x)
        pro1 = s1.predict_proba(test_x)
        pro2 = s2.predict_proba(test_x)
        print('遗传改进网络输出概率和分类以及准确率：')
        print(pro1)
        print(pre1_y)
        print(s1.score(test_x, test_y))
        print('实际分类')
        print(test_y)
        print('多层感知器输出概率和分类以及准确率：')
        print(pro2)
        print(pre2_y)
        print(s2.score(test_x, test_y))

    


        
        
        
        
