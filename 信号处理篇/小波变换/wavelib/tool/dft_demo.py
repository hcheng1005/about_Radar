import math
import cmath
import matplotlib.pyplot as plt
import numpy as np 

"""
计算一个序列的离散傅里叶变换

参数:
x (list): 输入序列

返回值:
X (list): 傅里叶变换结果
"""
def dft(x):

    N = len(x)
    X = []
    for k in range(N):
        re = 0.0
        im = 0.0
        for n in range(N):
            phi = 2 * math.pi * k * n / N
            re += x[n] * math.cos(phi)
            im -= x[n] * math.sin(phi)
        re = re / N
        im = im / N
        X.append(complex(re, im))
    return X


"""
计算一个序列的快速傅里叶变换

参数:
x (list): 输入序列

返回值:
X (list): 傅里叶变换结果
"""
def fft(x):
    N = len(x)
    print("N: ", N)
    if N <= 1:
        return x
    even = fft(x[0::2])
    
    # print("cal odd \n")
    
    odd =  fft(x[1::2])
    T= [cmath.exp(-2j*cmath.pi*k/N)*odd[k] for k in range(N//2)]
    return [even[k] + T[k] for k in range(N//2)] + [even[k] - T[k] for k in range(N//2)]

    

# 生成测试信号
N = 64
f1 = 5  # 第一个正弦信号的频率
f2 = 10 # 第二个正弦信号的频率
x = [math.sin(2 * math.pi * f1 * n / N) + math.sin(2 * math.pi * f2 * n / N) for n in range(N)]


plt.figure()
plt.subplot(121)
plt.plot(x)
# plt.show()

# 计算傅里叶变换
# X = dft(x)
X = fft(x)

dft_data = np.abs(np.array(X))
plt.subplot(122)
plt.plot(dft_data)
plt.show()
