
import numpy as np
import pywt
import matplotlib.pyplot as plt
from math import floor, ceil

def morlet_wavelet(t, s=1.0, w=5.0):
    """
    改进的Morlet小波函数。
    参数:
    - t : 时间数组
    - w : 频率参数
    - s : 尺度参数，影响小波的宽度
    """
    return (s**(-0.25)) * (np.pi**(-0.25)) * np.exp(-0.5 * t **2) * np.exp(1j * 2 * np.pi * w * t)
    # return np.exp(-0.5 * t ** 2) * np.exp(1j * 2 * np.pi * w * t)

def next_fast_len(n):
    """Round up size to the nearest power of two.

    Given a number of samples `n`, returns the next power of two
    following this number to take advantage of FFT speedup.
    This fallback is less efficient than `scipy.fftpack.next_fast_len`
    """
    return 2**ceil(np.log2(n))

def continuous_wavelet_transform(signal, scales, wavelet_function, dt=1.0, method='conv'):
    n = len(signal)
    coefficients = np.empty((np.size(scales),) + signal.shape, dtype=complex)

    signal = signal.astype(complex)
    
    # 初始化morlet
    x = np.linspace(-8, 8, 1024) # this param(-8,8) is same as pywt 
    step = x[1] - x[0]
    int_psi_scale = wavelet_function(x)
    int_psi = np.conj(int_psi_scale)
    int_psi = np.asarray(int_psi)
    
    if method == 'fft':
        size_scale0 = -1
        fft_data = None
    
    # plt.figure()
    # plt.ion()
    for i, scale in enumerate(scales):
        if 0: # 该方法是Pywavelet的实现方式
            j = np.arange(scale * (x[-1] - x[0]) + 1) / (scale * step)
            j = j.astype(int)  # floor
            if j[-1] >= int_psi.size:
                j = np.extract(j < int_psi.size, j)
            wavelet_data = int_psi[j][::-1]
        
        scaled_t =  (np.arange(scale * (x[-1] - x[0]) + 1) / scale - (x[-1] - x[0])*0.5)
        wavelet_data = wavelet_function(scaled_t, s=scale)
        
        # plt.clf()
        # plt.plot(wavelet_data)
        # plt.show()
        # plt.pause(0.05)
                
        if method == 'conv':            
            conv = np.convolve(signal, wavelet_data)
        else:
            # The padding is selected for:
            # - optimal FFT complexity
            # - to be larger than the two signals length to avoid circular
            #   convolution
            size_scale = next_fast_len(
                signal.shape[-1] + wavelet_data.size - 1
            )
            if size_scale != size_scale0:
                # Must recompute fft_data when the padding size changes.
                fft_data = np.fft.fft(signal, size_scale, axis=-1)
            size_scale0 = size_scale
            fft_wav = np.fft.fft(wavelet_data, size_scale, axis=-1)
            conv = np.fft.ifft(fft_wav * fft_data, axis=-1)
            conv = conv[..., :signal.shape[-1] + wavelet_data.size - 1]

        coef = - np.sqrt(scale) * np.diff(conv, axis=-1)
        # transform axis is always -1 due to the data reshape above
        d = (coef.shape[-1] - signal.shape[-1]) / 2.
        coef = coef[..., floor(d):-ceil(d)]
        coefficients[i, :] = coef
            
    return coefficients

# 创建一个测试信号
Fs = 512
t = np.linspace(0, 1, Fs, endpoint=False)
# signal = np.sin(2 * np.pi * 100 * t)  + np.sin(2 * np.pi * 50 * t)                                                # TEST SIGNAL 1
signal = np.hstack([np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 150 * t), np.sin(2 * np.pi * 200 * t) ])      # TEST SIGNAL 2

wavename = 'cmor5-5' # 5-5它们分别代表小波函数的参数，这些参数影响小波的频率和时间特性
# 定义缩放系数
totalscal = 100
# 中心频率
fc = pywt.central_frequency(wavename)
# 计算对应频率的小波尺度
cparam = 2 * fc * totalscal
scales = cparam / np.arange(totalscal, 1, -1)

# 计算CWT
coefficients1 = continuous_wavelet_transform(signal, scales, morlet_wavelet, method='fft')

# 使用pywavelet库进行小波变换
coefficients2, frequencies = pywt.cwt(signal, scales, wavename, sampling_period=1/Fs, method='fft')

t = np.linspace(0, 2, Fs * 2, endpoint=False)
# 绘制结果
plt.figure()
plt.subplot(121)
plt.pcolormesh(t, frequencies,  abs(coefficients1))
plt.ylabel(u"freq(Hz)")
plt.xlabel(u"time(s)")
plt.subplots_adjust(hspace=0.4)
plt.subplot(122)
plt.pcolormesh(t, frequencies, abs(coefficients2))
plt.ylabel(u"freq(Hz)")
plt.xlabel(u"time(s)")
plt.subplots_adjust(hspace=0.4)
plt.show()