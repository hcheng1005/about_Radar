%==========================================================================
% 比较 Capon 算法 与 空间 FFT 算法 的空间谱估计
%==========================================================================

clear; clc; close all;

%————————————————————————————
% 1. 参数设置
%————————————————————————————
c      = 3e8;            % 波速
f0     = 77e9;           % 载频
lambda = c/f0;           % 波长
M      = 4;              % 阵元数
d      = (0:M-1).' * (lambda/2);  % 阵元位置（列向量）

theta_true = [-20, 30];   % 真实目标角度 [deg]
K = numel(theta_true);    % 目标数

Nsnap = 200;              % 快数
SNR_dB = 20;              % 信噪比（dB）
SNR    = 10^(SNR_dB/10);

delta  = 1e-2;            % 对角加载因子

%————————————————————————————
% 2. 合成接收数据 X (M×Nsnap)
%————————————————————————————
% 构造 steering matrix A (M×K)
A = zeros(M, K);
for k = 1:K
    A(:,k) = exp(-1j*2*pi/lambda * d * sind(theta_true(k)));
end

% 随机源信号 (白噪声模型)
S = (randn(K, Nsnap) + 1j*randn(K, Nsnap)) / sqrt(2) * sqrt(SNR);

% 接收噪声
Noise = (randn(M, Nsnap) + 1j*randn(M, Nsnap)) / sqrt(2);

% 快快数据
X = A * S + Noise;

%% 此处将数据修改为真实采集数据
X = [98369.4+-51945.9j, 51698.8+-71447.3j, -45030.7+-76215.7j, -47271+-33421.3j]';
Nsnap = 1;
%————————————————————————————
% 3. 协方差矩阵 + 对角加载 + 逆
%————————————————————————————
R     = (X * X') / Nsnap;
R     = R + delta * trace(R)/M * eye(M);
R_inv = inv(R);

%————————————————————————————
% 4. Capon 谱估计
%————————————————————————————
angle_grid = -90:1:90;      % 扫描角度
Ng        = numel(angle_grid);

% 构造所有扫描角度下的 steering matrix (M×Ng)
A_grid = exp(-1j*2*pi/lambda * d * sind(angle_grid));

% 计算 Capon 谱： P = 1 ./ real(diag(A_grid' * R_inv * A_grid))
P_Capon = 1 ./ real( sum(conj(A_grid) .* (R_inv * A_grid), 1) );
P_Capon = P_Capon / max(P_Capon);     % 归一化
P_Capon_dB = 10*log10(P_Capon);

%————————————————————————————
% 5. 基于空间 FFT 的谱估计
%————————————————————————————
nfft_sp = 512;    % FFT 点数
% 对每个 snapshot 做空间 FFT，沿阵元维度(行)：
X_fft = fftshift( fft(X, nfft_sp, 1), 1 );  % 大小 nfft_sp × Nsnap

% 取平均功率谱
P_FFT = mean( abs(X_fft).^2, 2 );
P_FFT = P_FFT / max(P_FFT);                % 归一化
P_FFT_dB = 10*log10(P_FFT);

% 空间频率 u = [-0.5, …, +0.5)
u = ((0:nfft_sp-1)'/nfft_sp) - 0.5;
theta_fft = asind(2*u);    % % 正确的角度映射： sin(theta) = 2*u  => theta = asind(2*u)

%————————————————————————————
% 6. 绘图对比
%————————————————————————————
figure('Position',[100 100 700 400]);
plot(angle_grid, P_Capon_dB, 'b-', 'LineWidth',1.8); hold on;
plot(theta_fft,   P_FFT_dB,   'r--','LineWidth',1.4);
grid on;
xlabel('角度 (°)');
ylabel('归一化功率谱 (dB)');
title(sprintf('Capon vs 空间 FFT (SNR=%d dB, N_{snap}=%d)', SNR_dB, Nsnap));
xlim([-90 90]);

% 标出真实目标
for k = 1:K
    plot(theta_true(k), 0, 'kv','MarkerFaceColor','y','MarkerSize',8);
end

legend('Capon 谱','空间 FFT 谱','真实目标','Location','Best');
hold off;

%————————————————————————————
% 7. 小结
%————————————————————————————
% - Capon 算法分辨率高，但计算量大；  
% - 空间 FFT 算法（等价于 Bartlett 波束形成）计算快，但主瓣宽；  
% - 上图中黄色倒三角标示的是真实目标方向。