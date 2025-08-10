# 如何提高FFT后的精度

# 粗→精 两步走 Doppler 处理流程

下面介绍典型的“粗→精”两步走 Doppler 处理流程：  
1) 用短时 FFT 做频率搜索，粗估多普勒；  
2) 在粗估频率上做去载频＋长积分／细 FFT，以获得更高 SNR 和更细分辨率。

---

## 1. 系统框图

```text
回波信号 x[n] ──┬──> 分帧／窗函数 ──> N₁ 点 FFT ──> 峰值检测 ──> 粗估 Doppler \hat f  
                │  
                └──> 去载频 exp(−j2π \hat f n/Fs) ──> 长时域积分 or N₂ 点 FFT ──> 精 Doppler \hat f_{\rm fine}
```

---

## 2. 步骤详解

### 步骤1：粗 FFT 搜索

1. **分帧 & 加窗**  
   将连续采样 `x[n]` 按 N₁ 点切帧（可重叠或不重叠），对每帧乘窗（Hanning、Blackman…）。

2. **FFT**  
   对每帧做 N₁ 点 FFT，得到频谱  
   $$X[k],\quad k=0\ldots N_1-1.$$

3. **峰值检测**  
   找到最大幅值 bin `k₀`，粗估多普勒频率：  
   $$\hat f = \frac{k₀}{N_1}\,Fs,\quad  
     \Delta f_1 = \frac{Fs}{N_1}.$$

4. **多目标**  
   若有多目标，可同时选取多个峰值。

> **优点**：实时性好，能并行全带搜索  
> **缺点**：分辨率受 N₁ 限制

---

### 步骤2：去载频 ＋ 长积分／细 FFT

1. **去载频**  
   对跨多帧拼接的长时序列 `x[n]` 做相位旋转：  
   $$x_d[n] = x[n]\cdot e^{-j2\pi\hat f\,n/Fs}\,. $$
   这样目标成 DC，其它噪声／杂波仍在频带中。

2. **长时域积分**  
   - **方法A (相位累加)**  
     $$y = \sum_{n=0}^{N_{\rm int}-1} x_d[n]\,,\quad  
       \text{SNR gain}\approx N_{\rm int}.$$
   - **方法B (细 FFT)**  
     对长度 \(N_2\) 序列做 FFT：  
     $$\Delta f_2 = \frac{Fs}{N_2}\ll\Delta f_1,\quad N_2\gg N_1.$$
     通常取 \(N_2 = M\,N_1\)，即跨 \(M\) 帧拼接。

3. **精 Doppler**  
   - 若只需测速：  
     - 方法A 取相位/积分结果  
     - 方法B 取 FFT 峰值  
   - 理论提升约 \(20\log_{10}M\) dB SNR，分辨率提升至 \(\Delta f_2\)。

> **注意**  
> - 相干积累时间 \(T_p=N_2/Fs\) 受目标机动与相位噪声限制  
> - 方法A 得到的是一个复数，需用 `unwrap` 提取频率  
> - 方法B 可直观地看到小带宽谱形

---

## 3. 参数选取与性能

- **粗 FFT 长度 \(N_1\)**：带宽 = \(Fs\)，分辨率 \(\Delta f_1=Fs/N_1\)。  
- **窗函数**：降低旁瓣，抑制杂波/RFI。  
- **帧数 \(M\)**：\(N_2=M\,N_1\)，精分辨率 \(\Delta f_2=Fs/(M\,N_1)\)。  
- **相干时间** \(T_p=M\,N_1/Fs\) 要显著小于目标相位失步时间。

**示例**  
- \(Fs=1\) kHz，\(N_1=256\,\Rightarrow\Delta f_1\approx3.9\) Hz  
- 取 \(M=40\)，\(N_2=10240\)，\(\Delta f_2\approx0.1\) Hz，\(T_p\approx10.24\) s，增益≈16 dB  
- 若目标加速剧烈，可采用非相干分段累积再合并功率

---

## 4. 伪码示例

```matlab
clear; clc; close all;

%--- 公共参数
Fs       = 20e3;      % 采样率
N1       = 1024;      % 粗 FFT 长度
N2       = 1024;     % 总采样点数
A        = 1;        % 信号幅度
SNRin_dB = -1;       % 每点输入 SNR(dB)
sigma2   = A^2/10^(SNRin_dB/10)/2;  
M        = 2000;     % Monte-Carlo 重复次数

%--- fD 的分布区间（例如在 [120 150] Hz 均匀抽样）
fD_min = 10;
fD_max = 400;

% 预分配
errA   = zeros(M,1);
errB   = zeros(M,1);
fD_true = zeros(M,1);

% 时间向量
t1 = (0:N1-1)'/Fs;
t2 = (0:N2-1)'/Fs;

for m = 1:M
    % 1) 随机抽一个真实 Doppler
    fD        = fD_min + (fD_max-fD_min)*rand;
    fD_true(m)= fD;
    
    % 2) 生成含噪信号
    noise = sqrt(sigma2)*(randn(N2,1)+1j*randn(N2,1));
    x     = A*exp(1j*2*pi*fD*t2) + noise;
    
    % 3) 粗 FFT 搜索
    X1      = fft(x(1:N1), N1);
    [~,k0]  = max(abs(X1));
    f_hat   = (k0-1)/N1 * Fs;
    
    % 4) 去载频
    x_d     = x .* exp(-1j*2*pi*f_hat*t2);
    
    % 方法A：相位累加估残余
    y_sum   = sum(x_d);
    f_resA  = angle(y_sum)/(2*pi*(N2/Fs));
    fA      = f_hat + f_resA;
    
    % 方法B：细 FFT
    Ysh     = fftshift( fft(x_d, N2) );
    fvec    = (-N2/2:N2/2-1)*(Fs/N2);
    [~,k1]  = max(abs(Ysh));
    f_resB  = fvec(k1);
    fB      = f_hat + f_resB;
    
    % 5) 记录误差
    errA(m) = fA - fD;
    errB(m) = fB - fD;
end

% 统计结果
meanA = mean(errA);  stdA = std(errA);
meanB = mean(errB);  stdB = std(errB);

fprintf('方法A (相位法)：平均误差 = %+6.3f Hz, 标准差 = %6.3f Hz\n', meanA, stdA);
fprintf('方法B (细 FFT)：平均误差 = %+6.3f Hz, 标准差 = %6.3f Hz\n', meanB, stdB);

% 画误差分布
figure;
histogram(errA,50,'Normalization','pdf'); hold on
histogram(errB,50,'Normalization','pdf');
legend('方法A','方法B');
xlabel('估计误差 (Hz)'); ylabel('概率密度');
title('Monte Carlo 估计误差分布');
```

---

通过此流程，你可同时保留粗 FFT 的并行搜索能力，又能在目标带宽区做高 SNR 深度积累，获得极佳的多普勒估计精度与分辨率。





# 与ZoomFFT的区别

下面从原理和实现两方面比较一下“粗→精”两步走和 Zoom FFT 的区别：

---

## 1. 算法流程对比

| 特点           | 两步走（粗→精）                              | Zoom FFT                                     |
|---------------|-------------------------------------------|---------------------------------------------|
| 粗搜索         | 短时 FFT 全带搜索，找出峰值索引 k₀             | 通常也是先做一次宽带 FFT 或扫频确定感兴趣带   |
| 频移补偿／下变频 | 用估出的 f̂ 在时域上直接乘 e−j2πf̂n/Fs 去载频   | 在 DDC（数字下变频）中 Mixing：乘 e−j2πf₀n/Fs |
| 带限＋降采样   | *可选*（常不做），细 FFT 仍在原 Fs 下运行        | 必须：带通／低通滤波 + 降采样，缩小 Fs         |
| 细分辨率 FFT   | 对全带或已去载频的信号直接做长 FFT             | 在降采样后的低 Fs 下做较短 FFT                |
| 计算量         | N₁-point FFT + N₂-point FFT                  | N₁-point FFT + 滤波器(多相) + (N₂/D)-point FFT |
| 适用场景       | 目标测速、相干积累、在线粗→精测速               | 需要对多个窄带频点做高分辨率扫描时更高效      |

---

## 2. 关键区别

1. **降采样 vs. 不降采样**  
   - Zoom FFT：带限→降采样，大幅降低后续 FFT 点数和运算量  
   - 两步走：常直接在原 Fs 上做长 FFT（或直接积分），没有多级降采样

2. **滤波器设计**  
   - Zoom FFT：要先设计窄带低通／带通 FIR（或多相）滤波器，保证无混叠  
   - 两步走：去载频后直接积分或 FFT，若要降采样也需额外滤波器，但常省略

3. **硬件／实时性**  
   - Zoom FFT：多级多相滤波＋降采样，适合 FPGA／ASIC 实时窄带逐点扫描  
   - 两步走：FFT+相位旋转+大规模累加，软件/CPU 实现更简单

4. **灵活性**  
   - Zoom FFT：可一次“Zoom”多个子带，支持动态变化的目标带  
   - 两步走：适合锁定单一目标或少数几个频点后深度跟踪

5. **分辨率与增益**  
   - 分辨率均由后级 FFT 点数决定  
   - Zoom FFT 靠降采样后做相对更短的 FFT 达到同样分辨率  
   - 两步走通常用完整 N₂ 做 FFT 或积分，可获得最高 SNR 增益

---

## 3. 小结

- 如果你只需对一个或少数几个粗估出来的频点做深度积累／精估，两步走（粗 FFT → 去载频 → 长积分/细 FFT）最简洁。  
- 如果要对宽带内任意多个子带做高分辨率搜索，且对运算资源或存储有严格限制，则 Zoom FFT（下变频 + 滤波 + 降采样 + 窄带 FFT）更高效。




## 4. 伪码示例

```matlab
% Zoom‐FFT 处理流程示例（基于之前的 CW Doppler 仿真）
clear; clc; close all;

%% 1) 参数
Fs       = 1e3;          % 原始采样率 (Hz)
N        = 1024;         % 总采样点数
fD       = 134.2;        % 真实多普勒 (Hz)
A        = 1;            % 信号幅度
SNRin_dB = -1;           % 每点输入 SNR(dB)
sigma2   = A^2/(2*10^(SNRin_dB/10));

t = (0:N-1)'/Fs;
x = A*exp(1j*2*pi*fD*t) + sqrt(sigma2)*(randn(N,1)+1j*randn(N,1));

%% 2) 粗 FFT 搜索
N1   = 512;
X    = fft(x(1:N1), N1);
[~,k0] = max(abs(X));
f_hat = (k0-1)/N1 * Fs;    % 粗估频率

%% 3) Zoom FFT
% 3.1 下变频 (Mix to baseband)
x_mix = x .* exp(-1j*2*pi*f_hat * t);

% 3.2 带限滤波 + 降采样
D  = 8;                  % 降采样因子
L  = 64;                 % FIR 滤波器阶数
h  = fir1(L, 1/D);       % 归一化截止频率 = (Fs/2)/(Fs) * 2 = 1/D
x_f = filter(h, 1, x_mix);
x_d = x_f(1:D:end);      % 降采样
fs_z = Fs / D;           % Zoom 之后的采样率

% 3.3 窄带 FFT
Nz   = length(x_d);
Yz   = fftshift( fft(x_d, Nz) );
fvec = (-Nz/2:Nz/2-1) * (fs_z/Nz);

[~, kz]     = max(abs(Yz));
f_res_zoom  = fvec(kz);           % 残余微小偏移
f_zoom_est  = f_hat + f_res_zoom; % 最终 Zoom‐FFT 估计

%% 4) 结果输出
fprintf('真实 Doppler    = %.3f Hz\n', fD);
fprintf('粗 FFT 估计    = %.3f Hz\n', f_hat);
fprintf('Zoom FFT 估计 = %.3f Hz\n', f_zoom_est);

% 5) 画图对比
figure;
subplot(2,1,1);
plot((0:N1-1)*(Fs/N1), abs(X));
xlim([0 Fs/2]);
xlabel('频率 (Hz)'); ylabel('|X|');
title('Step1: 粗 FFT 频谱');

subplot(2,1,2);
plot(fvec, abs(Yz));
xlabel('频率 (Hz)'); ylabel('|Y_z|');
title('Step2: Zoom FFT（带限降采样后）');
xlim([-fs_z/2 fs_z/2]);
```

---