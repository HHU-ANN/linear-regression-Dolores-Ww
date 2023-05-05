# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

import numpy as np

# 加载数据
x_train = np.load('/data/exp02/x_train.npy')
y_train = np.load('/data/exp02/y_train.npy')

# 岭回归
class ridgeregression():
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def fit(self, x, y):
        # 最小二乘法求解岭回归系数
        eye_m = np.eye(x.shape[1])
        self.theta = np.linalg.inv(x.t @ x + self.alpha * eye_m) @ x.t @ y
        
    def predict(self, x):
        return x @ self.theta

ridge_reg = ridgeregression(alpha=1)
ridge_reg.fit(x_train, y_train)
ridge_pred = ridge_reg.predict(x_train)

# lasso回归
class lassoregression():
    def __init__(self, alpha=1.0, max_iter=1000, lr=0.01):
        self.alpha = alpha
        self.max_iter = max_iter
        self.lr = lr
    
    def fit(self, x, y):
        self.theta = np.zeros((x.shape[1], 1))
        
        for _ in range(self.max_iter):
            # 梯度下降法更新theta
            for j in range(x.shape[1]):
                if self.theta[j] > 0:
                    g = x[:,j:j+1].t @ (x @ self.theta - y) + self.alpha
                elif self.theta[j] < 0:
                    g = x[:,j:j+1].t @ (x @ self.theta - y) - self.alpha
                else:
                    g = x[:,j:j+1].t @ (x @ self.theta - y)
                
                self.theta[j] -= self.lr * g
            
    def predict(self, x):
        return x @ self.theta

lasso_reg = lassoregression(alpha=1, max_iter=1000, lr=0.01)
lasso_reg.fit(x_train, y_train)
lasso_pred = lasso_reg.predict(x_train)


def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
