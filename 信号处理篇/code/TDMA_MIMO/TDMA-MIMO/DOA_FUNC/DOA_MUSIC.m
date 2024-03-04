function [PoutMusic] = DOA_MUSIC(X, P, thetaGrids)
    % X: 输入信号 Channel * ChirpNum
    % P: 目标数目
    % PoutMusic: 输出功率谱
    
    M = size(X, 1); % 阵元数
    snap = size(X, 2); % 快拍数
    RX = X * X' / snap; % 协方差矩阵
    
    % 对协方差矩阵R进行特征值分解，并将特征值从到到小排列，其中第r+1到第N个特征值对应的特征向量就是噪声子空间的基
    [V, D] = eig(RX); % 特征值分解
    eig_value = real(diag(D)); % 提取特征值
    [B, I] = sort(eig_value, 'descend'); % 排序特征值
    EN = V(:, I(P+1:end)); % 提取噪声子空间
    
    PoutMusic = zeros(1, length(thetaGrids));
    
    for id = 1 : length(thetaGrids)
        atheta_vec = exp(1j * 2 * pi * [0:M-1]' * 1 / 2 * sind(thetaGrids(id))); % 导向矢量
        PoutMusic(id) = ((1 / (atheta_vec' * EN * EN' * atheta_vec))) ; % 功率谱计算
    end
end

    
    
   