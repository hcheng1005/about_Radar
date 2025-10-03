%% 步骤 1：读取数据并重排为 [快时间(ADC样点) × 慢时间(chirp索引)]  
clear;  
close all;  

% 读取 DCA1000 采集的原始数据（需用户提供的函数）  
[retVal] = readDCA1000_1('./demo.bin');  

% 全局参数（由读取函数或外部配置给出）  
global numChirps;          % chirp 总数（慢时间长度）  
global numADCSamples;      % 每个 chirp 的 ADC 采样点数（快时间长度）  

% 将4路接收天线数据重排为矩阵（每列为一个 chirp 的快时间采样）  
RX1data = reshape(retVal(1,:),numADCSamples,numChirps);   % RX1 数据  
RX2data = reshape(retVal(2,:),numADCSamples,numChirps);   % RX2  
RX3data = reshape(retVal(3,:),numADCSamples,numChirps);   % RX3  
RX4data = reshape(retVal(4,:),numADCSamples,numChirps);   % RX4  
% 说明：本脚本后续使用的是 RX1 的奇数 chirp（常用于 TDM-MIMO 中某一发射天线的子序列）  

%% 步骤 2：设定 FMCW/RF/采样相关参数（用于解释与后续换算）  
c=3.0e8;                  % 光速 (m/s)  
slope=60e12;              % FMCW 调频斜率 (Hz/s)  
Tc=50e-6;                 % chirp 周期 (s)  
B=slope*Tc;               % 调频带宽 (Hz)  
Fs=4e6;                   % ADC 采样率 (Hz) —— 这是快时间采样率  
f0=60.36e9;               % 起始频率 (Hz)  
lambda=c/f0;              % 波长 (m)  
d=lambda/2;               % 阵元间距 (m)  
frame=400;                % 拟处理的“帧数”（这里用来指代将要处理的慢时间点数）  
Tf=0.05;                  % 帧周期/慢时间采样周期 (s) —— 生命体征的采样周期  
N=1024;                   % 后续生命体征 FFT 点数（频谱分辨率取决于 N 和 Tf）  

% 提示：生命体征的采样率为 fs_vital = 1/Tf（例如 0.05 s -> 20 Hz）  

%% 步骤 3：对 RX1 的奇数 chirp 做快时间海明窗并进行 Range-FFT（距离维）  
% 目的：减少频谱泄漏，提取每个慢时间点的距离谱以便寻找人体回波峰  
range_win = hamming(numADCSamples); % 生成海明窗（与快时间长度一致）  

for k=1:1:frame  
    % 选取奇数编号 chirp（2*k-1），常用于 TDM-MIMO 中选择单个 TX 序列  
    din_win(:,k)=RX1data(:,2*k-1).*range_win; % 快时间加窗  
    datafft(:,k)=fft(din_win(:,k));           % 对快时间做 FFT，得到距离向谱  
end  

%% 步骤 4：在预设距离门内找到峰值 Range-bin（认为对应目标/人体回波）  
% 说明：通过门限 [rangeBinStartIndex, rangeBinEndIndex] 限定有效距离范围，  
% 避免近零/远距离杂波干扰。对每个慢时间索引 k，选取该门内幅值最大的 bin。  

rangeBinStartIndex=3; % 起始 bin（示例：约对应 0.1 m 分辨率时的 0.3 m 起）  
rangeBinEndIndex=10;  % 结束 bin  

for k=1:1:frame  
    for j=rangeBinStartIndex:1:numADCSamples   
        % 在指定门内寻找最大值所在 bin，将其复数幅度作为“目标距离的回波”  
        if(abs(datafft(j,k))==max(abs(datafft((rangeBinStartIndex:rangeBinEndIndex),k))))   
            data(:,k)=datafft(j,k);  
        end  
    end  
end  
% 注意（重要）：  
% - 当前实现把标量 datafft(j,k) 赋给了整列 data(:,k)，即列向量被同一标量填充。  
%   后续把 data(:,k) 当作一个复数而非向量使用会造成维度/运算问题。  
% - 更合理的做法是把该标量保存在一维序列中，如 target(k)=datafft(j,k)。  
% - 此处不改变原逻辑，仅加注释提示。  

%% 步骤 5：分离实部/虚部（为后续相位计算做准备）  
for k=1:frame  
    data_real(:,k)=real(data(:,k));  
    data_imag(:,k)=imag(data(:,k));  
end  

%% 步骤 6：计算相位并做相位展开（unwrap）  
% 物理意义：胸廓呼吸/心跳引起目标距离的微小变化 -> 回波相位的细微周期性变化  
% 正确做法一般是：phi = unwrap(angle(z)) 或 unwrap(atan2(imag,real))  
for k=1:frame  
    % 当前写法使用 atan(y/x) 且是“矩阵右除 /”，不是逐元素 ./，容易出错；  
    % 且 atan 无法正确处理四象限，相比之下 angle()/atan2() 更稳健。  
    signal_phase(:,k)=atan(data_imag(:,k)/data_real(:,k));  
end  

% 手工“相位展开”：若相邻帧相位差超过阈值，则整体加/减 pi 进行修正  
% 注：Matlab 内置 unwrap 更稳健；此处保留原有手工逻辑。  
for k=2:frame  
    diff=signal_phase(:,k)-signal_phase(:,k-1);  
    if diff>pi/2  
        signal_phase(:,(k:end))=signal_phase(:,(k:end))-pi;  
    elseif diff<-pi/2  
        signal_phase(:,(k:end))=signal_phase(:,(k:end))+pi;  
    end  
end  

%% 步骤 7：相位一阶差分（去除慢漂）与脉冲噪声去除  
% 一阶差分：抑制慢变化趋势（类似高通），突显周期性成分  
for k=1:frame-1  
    delta_phase(:,k)=signal_phase(:,k+1)-signal_phase(:,k);  
end  

% 3点脉冲噪声去除：若中点相对两侧点均出现同号大幅跳变，视为脉冲，改用线性插值  
thresh=0.8;  
for k=1:frame-3  
    phaseUsedComputation(:,k)=filter_RemoveImpulseNoise( ...  
        delta_phase(:,k),delta_phase(:,k+1),delta_phase(:,k+2),thresh);  
end  

% 时间轴（慢时间）：将样本索引映射到秒  
index=1:1:frame-3;  
index=index*Tf;  

%% 步骤 8：生命体征总带（0.1–2 Hz）带通滤波与时/频域展示  
% 目标：滤除 DC 漂移和高频噪声，保留呼吸+心跳的主频段  
filter_delta_phase=filter(bpf_vitalsign,phaseUsedComputation);  
vital_sign=filter_delta_phase;  

% 时域图：呼吸+心跳混合信号（生命体征总带）  
figure(1);  
plot(index,vital_sign);  
xlabel('Time(s)','FontWeight','bold');  
ylabel('Amplitude','FontWeight','bold');  
title('心肺信号','FontWeight','bold');  

% 对生命体征序列做 FFT（用于频谱质量评估与可视化）  
vital_sign_fft=fft(vital_sign,N);  

% 构建频率轴与单边幅度谱  
freq=(0:1:N/2)/Tf/N;  % 频率轴：f = k * (1/Tf)/N = k*fs_vital/N  
P2 = abs(vital_sign_fft/(N-1));         % 双边谱幅度（归一化）  
P1 = P2(1:N/2+1);                       % 单边谱（保留前半部分）  
P1(2:end-1) = 2*P1(2:end-1);            % 中间项 ×2（能量对称分布）  

% 频域图：心肺信号频谱（建议 xlim [0,2] 观察呼吸与低心跳成分）  
figure(2);  
plot(freq,P1);  
xlim([0,2]);  
xlabel('Frequency(Hz)','FontWeight','bold');  
ylabel('Amplitude','FontWeight','bold');  
title('心肺信号频谱图','FontWeight','bold');  

%% 步骤 9：呼吸带通滤波（常用 ~0.1–0.6 Hz）与时/频域展示  
% 通过 bpf_breathe（需用户定义）提取呼吸主频段  
filter_delta_phase_breathe=filter(bpf_breathe,phaseUsedComputation);  
breathe=filter_delta_phase_breathe;  

% 呼吸信号时域图  
figure(3);  
plot(index,breathe);  
xlabel('Time(s)','FontWeight','bold');  
ylabel('Amplitude','FontWeight','bold');  
title('呼吸信号','FontWeight','bold');  

% 呼吸信号频谱  
breathe_fft=fft(breathe,N);  
P2_breathe = abs(breathe_fft/(N-1));  
P1_breathe = P2_breathe(1:N/2+1);  
P1_breathe(2:end-1) = 2*P1_breathe(2:end-1);  

figure(4);  
plot(freq,P1_breathe);  
xlim([0,2]);  
xlabel('Frequency(Hz)','FontWeight','bold');  
ylabel('Amplitude','FontWeight','bold');  
title('呼吸信号频谱图','FontWeight','bold');  

%% 步骤 10：心跳带通滤波（常用 ~0.9–2 Hz）与时/频域展示  
% 通过 bpf_heart（需用户定义）提取心跳主频段  
filter_delta_phase_heart=filter(bpf_heart,phaseUsedComputation);  
heart=filter_delta_phase_heart;  

% 心跳信号时域图  
figure(5);  
plot(index,heart);  
xlabel('Time(s)','FontWeight','bold');  
ylabel('Amplitude','FontWeight','bold ');  
title('心跳信号','FontWeight','bold');  

% 心跳信号频谱  
heart_fft=fft(heart,N);  
P2_heart = abs(heart_fft/(N-1));  
P1_heart = P2_heart(1:N/2+1);  
P1_heart(2:end-1) = 2*P1_heart(2:end-1);  

%% 步骤 11：心跳谐波检测与基频增强（避免把 2×HR 当成 HR）  
% 原理：心跳波形常含二次谐波，若检测到 2:1 的谐波关系，则增强基频峰值  

% 在 0.9–2 Hz 频段内找基频峰；在 1.8–4 Hz 内找二次谐波峰  
[heart_peaks,heart_peaksnum]=findpeaks(P1_heart,0.9,2,N,Tf);    % 基频候选  
[heart_harmonic_peaks,heart_harmonic_peaksnum]=findpeaks(P1_heart,1.8,4,N,Tf); % 2倍谐波候选  

% 将峰位置从 bin 索引换算为 Hz：f = idx / (N*Tf)  
heart_peaks=heart_peaks/N/Tf;  
heart_harmonic_peaks=heart_harmonic_peaks/N/Tf;  

% 遍历峰集合（按列向量计数）  
[heart_peaks_row,heart_peaks_column]=size(heart_peaks);  
[heart_harmonic_peaks_row,heart_harmonic_peaks_column]=size(heart_harmonic_peaks);  

for i=1:heart_peaks_column  
    % 当该基频峰与全谱最大峰的差值小于 0.3（经验阈值）时，认为是强峰，进一步检测谐波  
    if max(P1_heart)-P1_heart(round(heart_peaks(i)*N*Tf)+1)<0.3   
        for j=1:heart_harmonic_peaks_column  
            % 若存在频率严格等于 2 倍的谐波峰（严格相等较苛刻，工程上可用容差）  
            if heart_harmonic_peaks(j)/heart_peaks(i)==2  
                % 将该基频峰幅度翻倍，增强其权重（抑制 2×HR 误判）  
                P1_heart(round(heart_peaks(i)*N*Tf)+1)=2*P1_heart(round(heart_peaks(i)*N*Tf)+1);  
            end  
        end  
    end  
end  

%% 步骤 12：心跳最终频谱图（含谐波一致性增强后的结果）  
figure(6);
plot(freq,P1_heart);
xlim([0,4]);
xlabel('Frequency(Hz)','FontWeight','bold');
ylabel('Amplitude','FontWeight','bold');
title('心跳信号频谱图','FontWeight','bold');