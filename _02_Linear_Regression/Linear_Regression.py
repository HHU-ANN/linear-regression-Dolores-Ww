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

    alpha = 0.1  # L1正则化的权重
    learning_rate = 0.01  # 学习率
    epochs = 1000  # 迭代次数
def cost_function(x, y, w, alpha):
    n_samples = len(y)
    cost = (1. / (2 * n_samples)) * np.sum((np.dot(x, w) - y) ** 2)
    cost += alpha * np.sum(np.abs(w))
    return cost

def lasso(x,y,w,alpha):
    N = x.shape[0]
    y_pred = np.dot(x, w)
    residuals = y_pred - y
    gradients = np.dot(x.T, residuals) / N
    l1_grad = (alpha * np.sign(w)) / N
    gradients += l1_grad
    return gradients
    pass

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
