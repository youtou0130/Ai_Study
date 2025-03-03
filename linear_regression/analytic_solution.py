import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
#回归，有监督的机器学习，X，y
X1 = 2*np.random.rand(100, 1)
X2 = 3*np.random.rand(100, 1)
#print(len(X1))
#print(X1)
#这里要模拟出来的数据y是代表真实的数据，
y = 5 + 4*X1 + 3*X2+ np.random.randn(100, 1)

X_b = np.c_[np.ones((100,1)), X1, X2]

a = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print(a)

X_new = np.array([[0,0],[2,3]])
X_new_b = np.c_[np.ones((2,1)), X_new]
print(X_new_b)
y_predict = X_new_b.dot(a)
print(y_predict)

plt.plot(X_new[:, 0],y_predict, 'r-')
plt.plot(X1,y,'b.')
plt.axis([0, 2, 0, 25])
plt.show()
