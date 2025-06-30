# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    # 推理不需要 softmax
    def predict(self, x):
        return np.dot(x, self.W)

    # 但是训练需要, 求损失函数需要把结果先规范化,再计算交叉熵误差
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()
# 损失函数
f = lambda w: net.loss(x, t)
# 求梯度
dW = numerical_gradient(f, net.W)

print(dW)
