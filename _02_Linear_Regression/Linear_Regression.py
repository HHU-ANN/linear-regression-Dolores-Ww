# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

import numpy as np

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y



X_train, y_train = read_data()

# 建立岭回归类
class RidgeRegression:
    def __init__(self, alpha):
        self.alpha = alpha
    
    def fit(self, X, y):
        n_features = np.shape(X)[1]
        # 添加正则项，防止过拟合
        self.weights = np.dot(np.linalg.inv(np.dot(X.T, X) + 
                               self.alpha*np.eye(n_features)), 
                               np.dot(X.T, y))
    
    def predict(self, X):
        y_pred = np.dot(X, self.weights)
        return y_pred

# 建立Lasso回归类
class LassoRegression:
    def __init__(self, alpha, learning_rate=0.0001, max_iterations=50):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        
    def fit(self, X, y):
        n_samples, n_features = np.shape(X)
        self.weights = np.zeros(n_features)
        self.current_weights = np.zeros(n_features)
        
        for i in range(self.max_iterations):
            y_pred = np.dot(X, self.weights)
            error = y - y_pred
            # LASSO的梯度下降中的L1函数的导数形式，即如果x>0取1否则-1
            self.current_weights[:] = self.weights[:]
            self.weights += self.learning_rate * (np.dot(X.T, error) - self.alpha * np.sign(self.weights).reshape((n_features,)))
            self.weights = np.where(self.weights * np.sign(self.current_weights) < 0, 0, self.weights)
    
    def predict(self, X):
        y_pred = np.dot(X, self.weights)
        return y_pred
        l1_penalty = 0.1 # L1正则化系数 
    
    def Loss_function(X, y, theta): 
        n_samples = len(X) 
        y_pred = X.dot(theta) 
        error = y_pred - y 
        mse_loss = (1/n_samples) * np.sum(error**2) 
        l1_loss = l1_penalty * np.sum(np.abs(theta)) 
        return mse_loss + l1_loss 
    
    def Gradient_function(X, y, theta): 
        n_samples = len(X) 
        y_pred = X.dot(theta) 
        error = y_pred - y 
        gradient = (2/n_samples) * X.T.dot(error) + l1_penalty * np.sign(theta) 
        return gradient 

# 进行岭回归
def ridge(data):
    ridge_reg = RidgeRegression(alpha=0.1) # 设置参数alpha
    ridge_reg.fit(X_train, y_train) # 使用训练数据拟合模型
    
    data = np.reshape(data, (1, -1)) # 将数据改为2D矩阵形式
    result = ridge_reg.predict(data) # 进行预测
    return float(result)

def lasso(data):
    learning_rate = 0.0000001  # 学习率
    n_iterations = 20  # 迭代次数
    theta = np.ones(6) 
    for i in range(n_iterations):
        gradient = Gradient_function(X_train, y_train, theta)
        theta = theta - learning_rate * gradient
        cost = Loss_function(X_train, y_train, theta)
    theta = theta.flatten()
    y=np.dot(data, theta)
    print(y)
    print("\n")
    print(data)
    return float(y)
