clc;clear all;close all;

fid = fopen('sst_nino3.dat', 'r');
data = textscan(fid, '%f');
fclose(fid);

s = data{1,1};

fs = 1000;
wavename='cmor3-3';
totalscal=44;
Fc=centfrq(wavename); % 小波的中心频率
c=2*Fc*totalscal;
scals=c./(1:totalscal);
f=scal2frq(scals,wavename,1/fs); % 将尺度转换为频率
coefs=cwt(s,'amor'); % 求连续小波系数


figure
imagesc(abs(coefs));
set(gca,'YDir','normal')
colorbar;
xlabel('时间 t/s');
ylabel('频率 f/Hz');
title('小波时频图');

figure
mesh(abs(coefs));
set(gca,'YDir','normal')
colorbar;
xlabel('时间 t/s');
ylabel('频率 f/Hz');
title('小波时频图');