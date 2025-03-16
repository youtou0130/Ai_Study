# 小批量梯度下降
import numpy as np

X = 2*np.random.rand(100, 1)
y = 4 + 3*X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]

#learning_rate = 0.0001
t0, t1 = 5, 500
# 定义一个函数来调整学习率
def learning_rate_schedule(t):
    return t0/(t+t1)

n_epochs = 10000
m = 100
batch_size = 10
num_batches = int(m / batch_size)
theta = np.random.randn(2, 1)

for epoch in range(n_epochs):
    # 在双层for循环之间，每个轮次开始迭代之前打乱数据索引顺序
    arr = np.arange(len(X_b))
    np.random.shuffle(arr)
    X_b = X_b[arr]
    y = y[arr]
    for i in range(num_batches):
        #random_index = np.random.randint(m)
        x_batch = X_b[i*batch_size:i*batch_size + batch_size]
        y_batch = y[i*batch_size:i*batch_size + batch_size]
        gradients = x_batch.T.dot(x_batch.dot(theta)-y_batch)
        learning_rate = learning_rate_schedule(epoch * m + i)
        theta = theta - learning_rate * gradients

print(theta)

