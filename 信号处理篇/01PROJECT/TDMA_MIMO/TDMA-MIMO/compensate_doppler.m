function [com_dopplerFFTOut] = compensate_doppler(doaInput, cfgOut, dopplerIdx, speedVal, rangeVal)
    %% 本文件用于生成由TDMA-MIMO导致的多普勒补偿
    %% By Xuliang, 20230418
    tarNum = length(dopplerIdx); % 目标数目
    numTx = cfgOut.numTx;
    numRx = cfgOut.numRx;
    ChirpNum = cfgOut.ChirpNum;
    angleFFT_size = 128;
%     dopplerIdx = rd_peak_list(2, :);

    if cfgOut.applyVmaxExtend 
        sig_bin = [];
        info_overlaped_diff1tx = cfgOut.virtual_array.info_overlaped_diff1tx; 
        if mod(numTx, 2) == 1 % 天线数目为奇数
            tmpDopplerIdx = reshape(dopplerIdx, [], 1); % 临时多普勒索引
            dopplerInd_unwrap = tmpDopplerIdx + ((1:numTx) - ceil(numTx / 2)) * ChirpNum; % tarNum * TXNum
        else % 天线数目为偶数
            tmpDopplerIdx = reshape(dopplerIdx, [], 1); % 临时多普勒索引
            if speedVal > 0
                dopplerInd_unwrap = tmpDopplerIdx + ((1:numTx)- (numTx / 2 + 1)) * ChirpNum; % tarNum * TXNum
            else
                dopplerInd_unwrap = tmpDopplerIdx + ((1:numTx)- numTx / 2) * ChirpNum; % tarNum * TXNum
            end
        end
        
        sig_bin_org = doaInput; % TARNUM * RXNUM * TXNUM
        deltaPhi = 2 * pi * (dopplerInd_unwrap - ChirpNum / 2) / (numTx * ChirpNum); % 多普勒相位修正 tarNum * TXNUM 
        deltaPhis(:, 1, 1, :) = deltaPhi; % tarNum * 1 * 1 * TXNUM
        tmpTX = (0 : numTx - 1);
        tmpTXs(1,1,:,1) = tmpTX; % tarNum * 1 * TXNUM
        correct_martrix = exp(-1j * tmpTXs .* deltaPhis); % tarNum * 1 * TXNUM * TXNUM
        sig_bin = sig_bin_org .* correct_martrix; % tarNum * RXNUM * TXNUM * TXNUM

        % 使用重叠天线来做最大速度的解包 
        nore_tx = [];
        nore_rx = [];
        re_tx = [];
        re_rx = [];
        for iid = 1 : length(info_overlaped_diff1tx)
            nore_rx_set = info_overlaped_diff1tx(:,3);
            nore_tx_set = info_overlaped_diff1tx(:,4);
            re_rx_set = info_overlaped_diff1tx(:,7);
            re_tx_set = info_overlaped_diff1tx(:,8);
            nore_rx = [nore_rx, find(cfgOut.PosRX_BOARD_ID == nore_rx_set(iid))];
            nore_tx = [nore_tx, find(cfgOut.PosTX_Trans_ID == nore_tx_set(iid))];
            re_rx = [re_rx, find(cfgOut.PosRX_BOARD_ID == re_rx_set(iid))];
            re_tx = [re_tx, find(cfgOut.PosTX_Trans_ID == re_tx_set(iid))];
        end
        index_overlaped_diff1tx = cat(2, nore_rx.', nore_tx.', re_rx.', re_tx.'); % 32 * 4
        
        % 获取未冗余阵元和冗余阵元对应的信号
        sig_overlap1 = zeros(tarNum, length(index_overlaped_diff1tx));
        sig_overlap2 = zeros(tarNum, length(index_overlaped_diff1tx));
        for iid = 1 : length(index_overlaped_diff1tx)
            tmppos1 = index_overlaped_diff1tx(:, 1);
            tmppos2 = index_overlaped_diff1tx(:, 2);
            tmppos3 = index_overlaped_diff1tx(:, 3);
            tmppos4 = index_overlaped_diff1tx(:, 4);
            sig_overlap1(:,iid) = sig_bin_org(:, tmppos1(iid), tmppos2(iid));
            sig_overlap2(:, iid) = sig_bin_org(:, tmppos1(iid), tmppos2(iid));
        end
        sig_overlap = cat(3, sig_overlap1, sig_overlap2); % tarNum * 32 * 2
        
        angle_sum_test = zeros(size(sig_overlap,1), size(sig_overlap,2), size(deltaPhi,2)); % 检查每个假设的重叠天线对相位差
        
        for sig_id = 1 : size(angle_sum_test,2)
            deltaPhiss(:, 1, :) = deltaPhi; % tarNum * 1 * TXnum
            tmpSigs = sig_overlap(:, 1:sig_id, 2); % tarNum * sig_id * 1
            signal2 = tmpSigs .* exp(-1j * deltaPhiss); % tarNum * 32 * TXnum
            angle_sum_test(:, sig_id, :) = angle(sum(sig_overlap(:,1:sig_id,1) .* conj(signal2),2)); % tarNum * 32 * TXnum
        end
        [~, doppler_unwrap_integ_overlap_index] = min(abs(angle_sum_test),[],3); % tarNum * 1 * 32
        doppler_unwrap_integ_overlap_index = squeeze(doppler_unwrap_integ_overlap_index); % 选择相位差最小的假设来估计解系数
        
        doppler_unwrap_integ_index = zeros(size(doppler_unwrap_integ_overlap_index, 1), 1);
        for i = 1:size(doppler_unwrap_integ_overlap_index, 1)
            doppler_unwrap_integ_index(i) = mode(doppler_unwrap_integ_overlap_index(i, :));
        end
        
        correct_sig = []; % tarNum * numRX * numTX
        for i = 1 : length(doppler_unwrap_integ_index)
            correct_sig(i, :, :) = squeeze(sig_bin(i, :, :, doppler_unwrap_integ_index(i))); 
        end
        
        index_noapply = find(rangeVal <= cfgOut.min_dis_apply_vmax_extend); % 未超出最小距离约束
        delta_phi_noapply = reshape(2 * pi * (dopplerIdx - ChirpNum / 2) / (numTx * ChirpNum),[],1);
        sig_bin_noapply = doaInput .* (reshape(exp(-1j * delta_phi_noapply * tmpTX), size(delta_phi_noapply, 1), 1, size(tmpTX,2))); % 单一相位约束
        correct_sig(index_noapply, :, :) = sig_bin_noapply(index_noapply, :, :);
        
        com_dopplerFFTOut = correct_sig;        
    else
        sig_bin_org = doaInput; % TARNUM * RXNUM * TXNUM
        % ves = (dopplerIdx - ChirpNum / 2) * 3e8 / cfgOut.fc / ( 2 * cfgOut.Tc * (numTx * ChirpNum));
        ves = doppler_(doaInput, dopplerIdx, cfgOut)
        deltaPhi = ves * 4 *pi * cfgOut.Tc * cfgOut.fc / 3e8;

        deltaPhi = deltaPhi.';
        tmpTX = (0 : numTx - 1); % 1 * TXNUM
        correct_martrix = exp(-1j * deltaPhi * tmpTX ); % TARNUM * TXNUM 
        correct_martrixs(:, 1, :) = correct_martrix;
        com_dopplerFFTOut = sig_bin_org .* correct_martrixs; % TARNUM * RXNUM * TXNUM
    end
end

%% 解算正确的多普勒速度
%% 参考文章： https://www.ti.com.cn/cn/lit/an/zhca901/zhca901.pdf
function ves = doppler_(doaInput, dopplerIdx, cfgOut)
    VMAX = (cfgOut.ChirpNum / 2 - 1) * 3e8 / cfgOut.fc / ( 2 * cfgOut.Tc * (cfgOut.numTx * cfgOut.ChirpNum))
    ves = zeros(1,length(dopplerIdx));
    sig_bin_org = doaInput;
    sig_bin_org = reshape(sig_bin_org, length(dopplerIdx),[]);
    sig_bin_org = sig_bin_org(1:length(dopplerIdx), [1,2,3,4,5,6,7,8]);
    for k=1:length(dopplerIdx)
        % 根据峰值索引计算多普勒速度
        vest = (dopplerIdx(k) - cfgOut.ChirpNum / 2) * 3e8 / cfgOut.fc / ( 2 * cfgOut.Tc * (cfgOut.numTx * cfgOut.ChirpNum));
        % 提取8RX相位信息
        angle_inf = sig_bin_org(k,:);  
        % 得到相位偏移           
        delta_Pfiest = vest * 4 *pi * cfgOut.Tc * cfgOut.fc / 3e8;
        j = sqrt(-1);
        angle_inf1 = angle_inf;
        % 相位补偿
        angle_inf1(5:8) = angle_inf1(5:8)*exp(-j*delta_Pfiest/2);  
        % 计算第一次角度峰值位置
        angle_result1 = abs(fftshift(fft(angle_inf1,128)));
        [max1,~] = max(angle_result1);
        
        % 反转符号后再次计算角度峰值位置
        angle_inf2 = angle_inf1; 
        angle_inf2(5:8) = angle_inf2(5:8)*-1; %对补偿后的TX2的Rx1-4进行符号反转
        angle_result2 = abs(fftshift(fft(angle_inf2,128)));
        [max2,~] = max(angle_result2);

        % 确定最终多普勒速度
        if max1>max2
            vreal = vest;
        else
            if vest>0
                vreal = vest-2*VMAX;
            else
                vreal = vest+2*VMAX;
            end
        end
        ves(k) = vreal;
    end
end