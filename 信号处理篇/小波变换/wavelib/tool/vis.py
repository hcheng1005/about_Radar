import numpy as np
import matplotlib.pyplot as plt

# 读取数据
data = np.loadtxt('/home/zdhjs-05/myGitHubCode/mycode/wavelib/test/sst_nino3.dat')
# print(data.shape)

plt.figure()
plt.subplot(1,2,1)
plt.plot(data)

cwt_result = np.loadtxt('/home/zdhjs-05/myGitHubCode/mycode/wavelib/build/cwt_result.dat')
cwt_result = cwt_result[:,2].reshape([22, 504])
# print(cwt_result.shape)

t = np.linspace(0, 1, 504)
freqs = np.arange(1, 23)

plt.subplot(1,2,2)
plt.contourf(t, freqs, abs(cwt_result), cmap='jet')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.colorbar()
plt.title('Continuous Wavelet Transform')

plt.tight_layout()
plt.show()

# 绘制小波变换的三维曲面图
# fig = plt.figure(figsize=(12, 6))
# ax2 = fig.add_subplot(1, 1, 1, projection='3d')
# X, Y = np.meshgrid(t, freqs)
# Z = abs(cwt_result)
# ax2.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet')
# ax2.set_xlabel('Time')
# ax2.set_ylabel('Frequency')
# ax2.set_zlabel('Wavelet Coefficient')
# ax2.set_title('Continuous Wavelet Transform')

# plt.tight_layout()
# plt.show()