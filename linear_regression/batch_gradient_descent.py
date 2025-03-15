#全量梯度下降
import numpy as np

#データセット生成X,y
np.random.seed(1) # 随机种子
X = np.random.rand(100, 1)
y = 4 + 3*X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]

#パラメータ生成
learning_rate = 0.001
n_iterations = 10000

# 1,初期化,theta,W0...Wn,
theta = np.random.randn(2, 1)

# 4,判断是否收敛，
# 一般不会去设定阈值，而是直接采用设置相对大的迭代次数保证可以收敛
for _ in range(n_iterations):
    # 2,求梯度,計算gradient
    gradient = X_b.T.dot(X_b.dot(theta)-y)
    # 3,应用梯度下降法公式去调整theta值 θ
    theta = theta - learning_rate * gradient

print(theta)


