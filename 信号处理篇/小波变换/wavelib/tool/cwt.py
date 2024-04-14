import numpy as np
import matplotlib.pyplot as plt

import pywt

# 读取数据
data = np.loadtxt('/home/zdhjs-05/myGitHubCode/mycode/wavelib/test/sst_nino3.dat')
print(data.shape)

plt.figure()
plt.subplot(1,2,1)
plt.plot(data)
# plt.show()

# 计算连续小波变换
scales = np.arange(1, 44)
coef, freqs = pywt.cwt(data, scales, 'morl')

t = np.linspace(0, 1, 504)
plt.subplot(1,2,2)
plt.contourf(t, freqs, abs(coef), cmap='jet')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.colorbar()
plt.title('Continuous Wavelet Transform')

plt.tight_layout()
plt.show()