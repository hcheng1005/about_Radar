function y = filter_RemoveImpulseNoise(dataPrev2, dataPrev1, dataCurr, thresh)
%FILTER_REMOVEIMPULSENOISE 去除一维序列中的脉冲型噪声（3点窗口）
%
% y = filter_RemoveImpulseNoise(dataPrev2, dataPrev1, dataCurr, thresh)
%
% 输入参数:
%   - dataPrev2 : 窗口内第1个样本（较早的点，例如 k）
%   - dataPrev1 : 窗口内第2个样本（中间点，例如 k+1，待判定是否为“脉冲”）
%   - dataCurr  : 窗口内第3个样本（较新的点，例如 k+2）
%   - thresh    : 判定阈值（与数据量纲一致，越大越不敏感；如 0.8）
%
% 输出参数:
%   - y         : 在该窗口位置输出的“去脉冲”结果：
%                 若中间点与两侧点的差分在同号方向且均超过阈值，
%                 则判定中间点为脉冲，输出为两侧点的线性插值；
%                 否则直接输出中间点。
%
% 使用场景:
%   - 典型用于相位差分序列的脉冲噪声剔除。滑动调用本函数：
%       y(k) = f(delta(k), delta(k+1), delta(k+2), thresh)
%   - 该函数为点态、无状态处理，复杂度 O(1)。
%
% 说明:
%   - “线性插值”在这里等价于 (左邻 + 右邻)/2，因为插值点位于两邻点的中点。
%   - 阈值应结合数据噪声水平设定；过小会误抑真实信号尖峰，过大会放过脉冲。
%   - 该规则要求“同号”且“双侧都大”，以避免把正常边缘或单侧突变误判为脉冲。

% 将3个输入样本放入局部数组，便于统一索引（可读性）
pDataIn = [];
pDataIn(1) = dataPrev2;  % 左邻点（较早）
pDataIn(2) = dataPrev1;  % 中间点（待判定）
pDataIn(3) = dataCurr;   % 右邻点（较新）

% 计算中间点相对左右邻点的差分
backwardDiff = pDataIn(2) - pDataIn(1); % 中间 - 左邻
forwardDiff  = pDataIn(2) - pDataIn(3); % 中间 - 右邻

% 线性插值的几何参数（可简化为 y = (y1 + y2)/2）
x1 = 0;                  % 左邻的“位置”
x2 = 2;                  % 右邻的“位置”
y1 = pDataIn(1);         % 左邻的数值
y2 = pDataIn(3);         % 右邻的数值
x  = 1;                  % 中点的位置（插值点）

% 脉冲判定条件：
% - 若中间点相对左右两点的差分同号且幅值均超过阈值，认为中间点是“尖突”
%   情况A：forwardDiff >  thresh 且 backwardDiff >  thresh  → 中间点显著偏高
%   情况B：forwardDiff < -thresh 且 backwardDiff < -thresh  → 中间点显著偏低
if ( (forwardDiff >  thresh && backwardDiff >  thresh) || ...
     (forwardDiff < -thresh && backwardDiff < -thresh) )

    % 用左右邻点对中点做线性插值（等价于 (y1 + y2)/2）
    % 写成通用线性插值形式，便于阅读和拓展
    y = y1 + ( (x - x1) * (y2 - y1) ) / (x2 - x1);

else
    % 否则认为中间点正常，直接保留
    y = pDataIn(2);
end
end