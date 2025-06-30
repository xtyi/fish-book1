# coding: utf-8
import numpy as np
import matplotlib.pylab as plt

# 数值微分
def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

# 函数
def function_1(x):
    return 0.01*x**2 + 0.1*x

# 计算切线
def tangent_line(f, x):
    # 返回 f 在 x 处的数值微分
    d = numerical_diff(f, x)
    print(d)
    # 计算 dx 与 f(x) 的差值
    y = f(x) - d*x
    # 返回根据数值微分得到的函数
    return lambda t: d*t + y

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function_1, 5)
y2 = tf(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.show()
