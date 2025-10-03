Fs = 20; N = 4; Fc1 = 0.1; Fc2 = 2.0;
Hd = designfilt('bandpassiir','FilterOrder',N, ...
    'HalfPowerFrequency1',Fc1,'HalfPowerFrequency2',Fc2, ...
    'SampleRate',Fs,'DesignMethod','butter');

% 幅度和相位/群时延响应
fvtool(Hd);             % 综合查看（幅频、相位、极零、群时延等）
% 或仅看群时延：
[gd,w] = grpdelay(Hd, 1024, Fs); plot(w, gd); grid on; xlabel('Hz'); ylabel('Group Delay (samples)');