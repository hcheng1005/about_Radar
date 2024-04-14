import numpy as np
import pywt
import matplotlib.pyplot as plt

# 创建一个测试信号：一个简单的正弦波叠加随机噪声
x = np.linspace(0, 1, num=512)
y = np.sin(2 * np.pi * 8 * x) + np.random.randn(x.size) * 0.5

# 选择小波基和分解层数
wavelet = 'db1'  # Daubechies小波
# max_level = pywt.dwt_max_level(len(y), pywt.Wavelet(wavelet))
max_level = 2
coeffs = pywt.wavedec(y, wavelet, level=max_level)

# 可视化近似系数和细节系数
plt.figure(figsize=(10, 6))
for i, coeff in enumerate(coeffs):
    levels = len(coeffs) - i
    if i == 0:
        plt.subplot(max_level+1, 1, i+1)
        plt.plot(coeff, label=f'Approximation Coefficients')
        plt.legend(loc='upper right')
    else:
        plt.subplot(max_level+1, 1, i+1)
        plt.plot(coeff, label=f'Detail Coefficients at Level {levels}')
        plt.legend(loc='upper right')
plt.tight_layout()
plt.show()