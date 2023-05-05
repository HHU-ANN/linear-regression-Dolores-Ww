# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(x,y,alpha):
    #添加截距列
    x = np.hstack((x,np.ones((len(x),1))))

    #加上L2正则项
    x_transpose = np.transpose(x)
    xTx_alphaI = np.dot(x_transpose,x) + alpha * np.eye(len(x[0]))
    inv_xTx_alphaI = np.linalg.inv(xTx_alphaI)

    #计算回归函数
    w = np.dot(np.dot(inv_xTx_alphaI,x_transpose),y)
    return w[:-1],w[-1]#返回系数和截距
    pass
    w, b = ridge(x, y, alpha=0.01)

    

def lasso(x,y,alpha=0.01, num_iterations=1000, tolerance=0.01):
    # 初始化权重向量w、偏置b和学习率learning_rate
    n_samples, n_features = x.shape
    w = np.random.rand(n_features)
    b = 0.
    learning_rate = 0.01

    for i in range(num_iterations):
        # 计算预测值和误差
        y_pred = x.dot(w) + b
        error = y - y_pred

        # 计算梯度
        dw = (x.T.dot(error) - alpha * np.sign(w)) / n_samples
        db = -np.sum(error) / n_samples

        # 更新权重和偏置
        w -= learning_rate * dw
        b -= learning_rate * db

        # 判断是否收敛
        if np.linalg.norm(dw, ord=1) < tolerance:
            break

    return w
    pass

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
