import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import numpy as np

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


class CNNRegressor:
    
    """
    初始化的输入参数convo_para是卷积池化层的参数,里面的每一个值，
    是一个列表[con_fiter,con_strides,pool_filter,pool_strides]
    其中的filter是[channels,heights,widths,deepth],strides是向各方向移动的步长
    """

    def __init__(self, convo_para=[
        [[10, 3, 3, 1], [1, 1, 1, 1], [1, 3, 3, 1], [1, 2, 2, 1]],
        [[5, 3, 3, 1], [1, 1, 1, 1], [1, 3, 3, 1], [1, 2, 2, 1]],
    ], hidden_layer_size=50, max_iter=100,times=0.2):
        self.batch_size = 8
        self.middle_layers_size = len(convo_para)
        self.conv_filters_sizes = [convo_para[i][0] for i in range(self.middle_layers_size)]
        self.conv_strides = [convo_para[i][1] for i in range(self.middle_layers_size)]
        self.pool_filters_sizes = [convo_para[i][2] for i in range(self.middle_layers_size)]
        self.pool_strides = [convo_para[i][3] for i in range(self.middle_layers_size)]
        self.hidden_layer_size = hidden_layer_size
        self.max_iter = max_iter
        self.times = times

    def generatebatch(self, X, Y, n_examples, batch_size):
        for batch_i in range(n_examples // batch_size):
            start = batch_i*batch_size
            end = start + batch_size
            batch_xs = X[start:end]
            batch_ys = Y[start:end]
            yield batch_xs, batch_ys  # 生成每一个batch

    def _fit(self, X, Y):
        """
        :param X: X是（batch, height, width, channels）数据，n是样本个数
        :param Y: Y是(batch, results)数据
        :return:
        """
        batch, input_height, input_width, input_deepth = X.shape
        tf.reset_default_graph()
        # 输入层
        tf_X = tf.placeholder(tf.float32, [None, input_height, input_width,  input_deepth])
        tf_Y = tf.placeholder(tf.float32, [None, Y.shape[1]])

        self.output_size = Y.shape[1]

        conv_X = tf_X

        self.conv_filter_w, self.conv_filter_b = [], []
        self.batch_mean, self.batch_var = [], []
        self.shift, self.scale = [], []

        for i in range(self.middle_layers_size):
            if i != 0:
                w_size = self.conv_filters_sizes[i][1:3] + [self.conv_filters_sizes[i-1][0]] + [self.conv_filters_sizes[i][0]]
            else:
                w_size = self.conv_filters_sizes[i][1:] + [self.conv_filters_sizes[i][0]]
            self.conv_filter_w.append(tf.Variable(tf.random_normal(w_size)))
            self.conv_filter_b.append(tf.Variable(tf.random_normal([self.conv_filters_sizes[i][0]])))

            # 卷积层
            conv_out = tf.nn.conv2d(conv_X, self.conv_filter_w[i], strides=self.conv_strides[i], padding='SAME') + self.conv_filter_b[i]

            # BN层
            if i!=0:
                batch_mean, batch_var = tf.nn.moments(conv_out, [0, 1, 2], keep_dims=True)
                self.batch_mean.append(batch_mean)
                self.batch_var.append(batch_var)
                self.shift.append(tf.Variable(tf.zeros([self.conv_filters_sizes[i][0]])))
                self.scale.append(tf.Variable(tf.ones([self.conv_filters_sizes[i][0]])))
                epsilon = 1e-3
                BN_out = tf.nn.batch_normalization(conv_out, self.batch_mean[i-1], self.batch_var[i-1], self.shift[i-1], self.scale[i-1], epsilon)
            else:
                BN_out = conv_out

            # 激活层
            relu_feature_maps = tf.nn.relu(BN_out)

            # 池化层
            max_pool = tf.nn.max_pool(relu_feature_maps, ksize=self.pool_filters_sizes[i], strides=self.pool_strides[i], padding='SAME')

            conv_X = max_pool

        input_size = max_pool.shape[1]*max_pool.shape[2]*max_pool.shape[3]
        max_pool_flat = tf.reshape(max_pool, [-1, input_size])

        # 全连接层
        self.fc_w = tf.Variable(tf.random_normal([input_size, self.hidden_layer_size]))
        self.fc_b = tf.Variable(tf.random_normal([self.hidden_layer_size]))
        fc_out1 = tf.nn.relu(tf.matmul(max_pool_flat, self.fc_w) + self.fc_b)

        # 输出层
        self.out_w = tf.Variable(tf.random_normal([self.hidden_layer_size, self.output_size]))
        self.out_b = tf.Variable(tf.random_normal([self.output_size]))
        pred = tf.nn.softmax(tf.matmul(fc_out1, self.out_w)+self.out_b)

        loss = -tf.reduce_mean(tf_Y*tf.log(tf.clip_by_value(pred, 1e-11, 1.0)))
        train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.max_iter):
                for batch_xs, batch_ys in self.generatebatch(X, Y, Y.shape[0], self.batch_size): # 每个周期进行MBGD算法
                    sess.run(train_step, feed_dict={tf_X: batch_xs, tf_Y: batch_ys})


            # 运算出神经网络参数
            for i in range(self.middle_layers_size):
                self.conv_filter_w[i] = sess.run(self.conv_filter_w[i])
                self.conv_filter_b[i] = sess.run(self.conv_filter_b[i])
                if i != 0:
                    self.batch_mean[i - 1] = sess.run(self.batch_mean[i - 1], feed_dict={tf_X: X, tf_Y: Y})
                    self.batch_var[i - 1] = sess.run(self.batch_var[i - 1], feed_dict={tf_X: X, tf_Y: Y})
                    self.shift[i - 1] = sess.run(self.shift[i - 1], feed_dict={tf_X: X, tf_Y: Y})
                    self.scale[i - 1] = sess.run(self.scale[i - 1], feed_dict={tf_X: X, tf_Y: Y})
            self.fc_w, self.fc_b = sess.run(self.fc_w), sess.run(self.fc_b)
            self.out_w, self.out_b = sess.run(self.out_w), sess.run(self.out_b)


    def _predict(self, X):

        tf.reset_default_graph()
        x = tf.placeholder(tf.float32, X.shape)
        conv_X = x
        for i in range(self.middle_layers_size):
            # 卷积层
            conv_out = tf.nn.conv2d(conv_X, self.conv_filter_w[i], strides=self.conv_strides[i], padding='SAME') + \
                       self.conv_filter_b[i]

            # BN层
            if i != 0:
                epsilon = 1e-3
                BN_out = tf.nn.batch_normalization(conv_out, self.batch_mean[i-1], self.batch_var[i-1], self.shift[i-1], self.scale[i-1], epsilon)
            else:
                BN_out = conv_out

            # 激活层
            relu_feature_maps = tf.nn.relu(BN_out)

            # 池化层
            max_pool = tf.nn.max_pool(relu_feature_maps, ksize=self.pool_filters_sizes[i], strides=self.pool_strides[i],
                                      padding='SAME')

            conv_X = max_pool

        input_size = max_pool.shape[1] * max_pool.shape[2] * max_pool.shape[3]
        max_pool_flat = tf.reshape(max_pool, [-1, input_size])

        # 全连接层
        fc_out1 = tf.nn.relu(tf.matmul(max_pool_flat, self.fc_w) + self.fc_b)

        # 输出层
        pred = tf.nn.softmax(tf.matmul(fc_out1, self.out_w) + self.out_b)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output = sess.run(pred, feed_dict={x: X})

        return output


    def fit(self, X, Y):
        """
        :param X: X是（n, height, width, deepth）数据，n是样本个数
        :param Y: Y是(n, results)数据
        :return:
        """
        self.x_shape = X.shape
        std_x, self.scale_x = self.standard(X.reshape([self.x_shape[0],-1]))
        std_y, self.scale_y = self.standard(Y)
        self._fit(std_x.reshape(self.x_shape), std_y)


    def predict(self, X):
        shape = [-1] + list(self.x_shape[1:])
        std_x = self.standard(X.reshape([-1,shape[1]*shape[2]*shape[3]]), scale=self.scale_x )
        output = self._predict(std_x.reshape(shape))
        pre = self.unstandard(output, self.scale_y)
        return pre

    def score(self, X, Y):
        pred_y = self.predict(X)
        if len(Y.shape) == 1:
            pred_y.shape = Y.shape
        u = np.sum(np.power(pred_y - Y, 2))
        v = np.sum(np.power(pred_y - np.mean(pred_y, axis=0), 2))
        return 1 - u / v

    def standard(self, data, scale=0):
        if scale==0:
            a = np.min(data, axis=0)
            b = np.max(data, axis=0)-a+1e-10
            a = a - self.times*b
            b = (2*self.times+1)*b
            std_data=(data-a)/b
            scale_data=[a,b]
            return std_data,scale_data
        else:
            std_data=(data-scale[0])/(scale[1])
            return std_data

    def unstandard(self,std_data,scale_data):
        a=scale_data[0]
        b=scale_data[1]
        data = std_data*b+a
        return data


class CNNClassifier(CNNRegressor):
    def __init__(self, convo_para=[
        [[10, 3, 3, 1], [1, 1, 1, 1], [1, 3, 3, 1], [1, 2, 2, 1]],
        [[5, 3, 3, 1], [1, 1, 1, 1], [1, 3, 3, 1], [1, 2, 2, 1]],
    ], hidden_layer_size=50, max_iter=100,times=0.2):
        super().__init__(convo_para=convo_para, hidden_layer_size=hidden_layer_size, max_iter=max_iter, times=times)

    def fit(self, X, Y):
        self.x_shape = X.shape
        y = OneHotEncoder().fit_transform(Y).todense()
        std_x, self.scale_x = self.standard(X.reshape([self.x_shape[0],-1]))
        self._fit(std_x.reshape(self.x_shape), y)


    def predict(self, X):
        shape = [-1] + list(self.x_shape[1:])
        std_x = self.standard(X.reshape([-1,shape[1]*shape[2]*shape[3]]), self.scale_x)
        pro = self._predict(std_x.reshape(shape))
        pre = np.argmax(pro, axis=1)
        return pre

    def score(self, X, Y):
        pre = self.predict(X)
        if (Y.ndim==2) and (1 in Y.shape):
            real_y = Y.reshape(-1)
        else:
           real_y = Y
        right = np.equal(pre, real_y).sum()
        return right / len(Y)



if __name__ == '__main__':
    digits = load_digits()
    X_data = digits.data.astype(np.float32)
    Y_data = digits.target.astype(np.float32).reshape(-1, 1)
    # Y = OneHotEncoder().fit_transform(Y_data) # one-hot编码
    X = X_data.reshape(-1, 8, 8, 1)
    cnn = CNNClassifier(max_iter=100)
    cnn.fit(X, Y_data)
    print(cnn.score(X, Y_data))
