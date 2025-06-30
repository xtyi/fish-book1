# coding: utf-8
import numpy as np
import matplotlib.pylab as plt
from gradient_2d import numerical_gradient

# 返回的 x 为最后一次点的坐标, x_history 是优化过程中所有时刻点的坐标
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append( x.copy() )

        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)

# 损失函数
def function_2(x):
    return x[0]**2 + x[1]**2

# 初始值
init_x = np.array([-3.0, 4.0])

# 学习率
lr = 0.1
# 过大的学习率
# lr = 0.8
# 过小的学习率
# lr = 1e-10
# 优化轮次
step_num = 20

x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

plt.plot( [-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()
