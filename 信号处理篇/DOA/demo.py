import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

def generate_signal(d, N, theta, SNR):
    """
    生成阵列信号。
    d: 阵元间距
    N: 阵元数量
    theta: 信号到达角度
    SNR: 信噪比
    """
    wavelength = 3e8/77e9  # 波长
    # wavelength = 1  # 波长
    k = 2 * np.pi / wavelength  # 波数
    angles = np.radians(theta)
    steering_vector = np.exp(-1j * k * d * np.arange(N)[:, np.newaxis] * np.sin(angles))
    signal = np.dot(steering_vector, np.random.randn(len(theta), 1))
    # noise = np.sqrt(0.5 / SNR) * (np.random.randn(N, 1) + 1j * np.random.randn(N, 1))
    noise = 0
    return signal + noise

def music_algorithm(array_data, N, d, num_sources):
    """
    MUSIC算法实现。
    array_data: 阵列接收的数据
    N: 阵元数量
    d: 阵元间距
    num_sources: 信号源数量
    """
    R = array_data @ array_data.conj().T / array_data.shape[1]  # 计算协方差矩阵
    eigenvalues, eigenvectors = eigh(R)
    noise_subspace = eigenvectors[:, :-num_sources]

    angles = np.linspace(  -90, 90, 256)
    spectrum = np.zeros(angles.shape)
    for i, angle in enumerate(angles):
        steering_vector = np.exp(-1j * 2 * np.pi * d * np.arange(N) * np.sin(np.radians(angle)) / 1)
        spectrum[i] = 1 / np.abs(steering_vector.conj().T @ noise_subspace @ noise_subspace.conj().T @ steering_vector)

    return angles, spectrum

# 参数设置
N = 12 * 14  # 阵元数量
d = 0.5 # 阵元间距
theta = [20]  # 信号到达角度
SNR = 30  # 信噪比

# 生成和处理信号
array_data = generate_signal(d, N, theta, SNR)
print(array_data.shape)

angles, spectrum = music_algorithm(array_data, N, d, len(theta))
'''
description: 
return {*}
'''
# 直接使用FFT
fft_spectrum = np.abs(np.fft.fftshift(np.fft.fft(array_data[:,0], 256)))


# 绘制结果
plt.figure(figsize=(20,10))
plt.subplot(121)
plt.plot(np.real(array_data))
plt.subplot(122)
plt.plot(angles, 10 * np.log10(spectrum))
plt.plot(angles, (fft_spectrum))
plt.title('MUSIC Spectrum')
plt.xlabel('Angle (degrees)')
plt.ylabel('Spatial Spectrum (dB)')
plt.grid(True)
plt.show()
