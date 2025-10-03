%% IQ imbalance compensation demo
% 功能:
% 1) 构造带有 I/Q 不平衡与直流偏置的测试信号
% 2) 采用二阶统计量估计幅度失配 (epsilon) 与相位失配 (phi)
% 3) 去直流并构建 2x2 线性校正矩阵进行补偿
% 4) 对比补偿前后：时域波形、IQ 相图（星座/利萨如）、频谱（镜像抑制）

clc; clear; close all;

%% -----------------------------
% 参数配置
% ------------------------------
fs  = 20e6;          % 采样率 [Hz]
fc  = 2.5e6;         % 基带测试音频率（可理解为基带单音）[Hz]
T   = 1e-4;          % 信号时长 [s]，样本数越多统计越稳定
t   = 0:1/fs:T-1/fs; % 时间向量

% 设定 I/Q 不平衡与直流偏置（用于模拟）
phi_deg = 10;        % 相位失配（度）。I/Q 偏离 90° 的误差量
epsilon  = 0.3;      % 幅度失配因子（Q 相对 I 的增益误差），例如 0.3 => Q 增益大 30%
di       = 0.2;      % I 路直流偏置
dq       = -0.15;    % Q 路直流偏置

% 随机噪声级（可调，用于更接近实测）
snr_dB   = 40;       % 加性高斯白噪声的 SNR

%% -----------------------------
% 构造理想与失衡 I/Q 信号
% ------------------------------
% 理想的基带 I/Q（以同频正交余弦为例）
I_ideal = cos(2*pi*fc*t);
Q_ideal = sin(2*pi*fc*t); % 与 I 理想正交

% 引入幅度失配 + 相位失配 + 直流偏置
phi_rad = phi_deg*pi/180;              % 转换为弧度
I_raw   = I_ideal + di;                % I 路加 DC
Q_raw   = (1+epsilon)*sin(2*pi*fc*t + phi_rad) + dq; % Q 路增益与相位失配并加 DC

% 合成为复基带（工程习惯：s = I + jQ）
x_raw = I_raw + 1j*Q_raw;

% 加性噪声（可选）
x_noisy = awgn(x_raw, snr_dB, 'measured');
I_meas  = real(x_noisy);
Q_meas  = imag(x_noisy);

%% -----------------------------
% 1) 去直流偏置
% ------------------------------
% 估计直流均值（使用测量的 noisy 信号）
di_hat = mean(I_meas);
dq_hat = mean(Q_meas);

% 去 DC 后的 I/Q
I0 = I_meas - di_hat;
Q0 = Q_meas - dq_hat;

%% -----------------------------
% 2) 估计幅度与相位失配 (epsilon, phi)
%    基于二阶统计量的无监督估计
%    假设目标信号满足: E{I'Q'}=0, E{I'^2}=E{Q'^2}
% ------------------------------
EII = mean(I0 .* I0);
EQQ = mean(Q0 .* Q0);
EIQ = mean(I0 .* Q0);

% epsilon 估计：相对功率差的平方根
epsilon_hat = sqrt(max(EQQ, eps)/max(EII, eps)) - 1;

% phi 估计：由相关系数决定（取反正弦）
rho = EIQ / sqrt(max(EII*EQQ, eps));
rho = max(min(rho, 1-1e-12), -1+1e-12); % 数值夹紧，避免超出定义域
phi_hat = -asin(rho);                    % [rad]

%% -----------------------------
% 3) 构造校正矩阵并补偿
%    P = [ 1,                 0;
%          tan(phi),  1/((1+epsilon)*cos(phi)) ]
% 作用：对 Q 做去耦合与等幅处理，使 I/Q 重新正交、等功率
% 注：先去 DC，再线性变换
% ------------------------------
P = [ 1,                 0;
      tan(phi_hat),  1/((1+epsilon_hat)*cos(phi_hat)) ];

IQ0    = [I0; Q0];
IQcorr = P * IQ0;   % 应用线性校正

I_corr = IQcorr(1, :);
Q_corr = IQcorr(2, :);
x_corr = I_corr + 1j*Q_corr;

%% -----------------------------
% 4) 评估与可视化
%    新增：补偿前后 IQ 对比图（时域与相图）
%    频谱对比：观察镜像抑制
% ------------------------------

% 4.1 打印估计结果
fprintf('--- Estimated imbalance ---\n');
fprintf('epsilon_hat = %.4f (true: %.4f)\n', epsilon_hat, epsilon);
fprintf('phi_hat     = %.4f rad (%.2f deg), true: %.4f rad (%.2f deg)\n', ...
        phi_hat, phi_hat*180/pi, phi_rad, phi_deg);

% 4.2 补偿前后的时域波形对比（I/Q 分别绘制）
Nshow = min(400, numel(t)); % 仅展示前 Nshow 点，图像更清晰

figure('Name','Time-domain I/Q comparison','Color','w','Position',[100 100 1200 600]);
subplot(2,1,1);
plot(t(1:Nshow)*1e6, I_meas(1:Nshow), 'r-', 'LineWidth', 1); hold on;
plot(t(1:Nshow)*1e6, Q_meas(1:Nshow), 'b-', 'LineWidth', 1);
grid on; xlabel('Time [\mus]'); ylabel('Amplitude');
title('I channel: before vs after correction');
legend('Before (I_{meas})','After (I_{corr})');

subplot(2,1,2);
plot(t(1:Nshow)*1e6, I_corr(1:Nshow), 'r-', 'LineWidth', 1); hold on;
plot(t(1:Nshow)*1e6, Q_corr(1:Nshow), 'b-', 'LineWidth', 1);
grid on; xlabel('Time [\mus]'); ylabel('Amplitude');
title('Q channel: before vs after correction');
legend('Before (Q_{meas})','After (Q_{corr})');

% 4.3 补偿前后的 IQ 相图（利萨如/星座风格）
% 为了更清晰，随机抽样一部分点
rng(1);
idx = randperm(numel(I_meas), min(20000, numel(I_meas)));

figure('Name','IQ Lissajous / Constellation comparison','Color','w','Position',[200 200 1200 500]);
subplot(1,2,1);
plot(I0(idx), Q0(idx), '.', 'Color',[0.85 0.2 0.2], 'MarkerSize', 4); grid on; axis equal;
xlabel('I_0'); ylabel('Q_0'); title('Before correction (after DC removal)');
% 椭圆/剪切形状意味着存在不平衡
subplot(1,2,2);
plot(I_corr(idx), Q_corr(idx), '.', 'Color',[0.2 0.3 0.9], 'MarkerSize', 4); grid on; axis equal;
xlabel('I_{corr}'); ylabel('Q_{corr}'); title('After correction');
% 期望点云更接近圆形/标准正交

% 4.4 频谱对比（Welch PSD，观察镜像分量）
nfft = 4096;
figure('Name','Spectrum (Welch PSD) comparison','Color','w','Position',[150 150 1200 500]);
subplot(1,2,1);
pwelch(x_noisy, hamming(1024), 512, nfft, fs, 'centered'); grid on;
title('Before correction (centered PSD)');
ylabel('PSD [dB/Hz]');

subplot(1,2,2);
pwelch(x_corr,  hamming(1024), 512, nfft, fs, 'centered'); grid on;
title('After correction (centered PSD)');
ylabel('PSD [dB/Hz]');

%% -----------------------------
% 5) 简单数值验证：统计特性是否恢复
% ------------------------------
EII_corr = mean(I_corr.*I_corr);
EQQ_corr = mean(Q_corr.*Q_corr);
EIQ_corr = mean(I_corr.*Q_corr);
IMRR_est_before = 10*log10( (abs(EII)+abs(EQQ)) / (2*abs(EIQ)+eps) ); % 粗略镜像抑制 proxy
IMRR_est_after  = 10*log10( (abs(EII_corr)+abs(EQQ_corr)) / (2*abs(EIQ_corr)+eps) );

fprintf('--- Statistical checks ---\n');
fprintf('Before: E{I0*Q0}=%.3e, E{I0^2}=%.3e, E{Q0^2}=%.3e\n', EIQ, EII, EQQ);
fprintf('After : E{Ic*Qc}=%.3e, E{Ic^2}=%.3e, E{Qc^2}=%.3e\n', EIQ_corr, EII_corr, EQQ_corr);
fprintf('Proxy IMRR [dB]  -> Before: %.2f  |  After: %.2f\n', IMRR_est_before, IMRR_est_after);

%% 结束
% 说明：
% - 上述 IMRR_est_* 是基于相关项的粗略 proxy，只用于演示改善程度。
% - 实际系统可用更准确的镜像功率测量方法（例如频域在镜像带积分）。