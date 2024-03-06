
clear; close all; clc
dbstop if error

%Choose object detection probability
P_D = 0.3;

%Choose clutter rate
lambda_c = 10;

%Choose linear or nonlinear scenario
scenario_type = 'linear';

% boolean value indicating whether to generate noisy object state sequence or not 
ifnoisy = 0;

%Create tracking scenario
switch(scenario_type)
    case 'linear'
        %Creat sensor model
        range_c = [-1000 1000;-1000 1000];
        sensor_model = modelgen.sensormodel(P_D,lambda_c,range_c);
        
        %Creat linear motion model
        T = 0.5;
        sigma_q = 0.5;
        motion_model = motionmodel.cvmodel(T,sigma_q);
        
        %Create linear measurement model
        sigma_r = 10;
        meas_model = measmodel.cvmeasmodel(sigma_r);
        
        %Creat ground truth model
        nbirths = 5;
        K = 100;
        tbirth = zeros(nbirths,1);
        tdeath = zeros(nbirths,1);
        
        initial_state = repmat(struct('x',[],'P',eye(motion_model.d)),[1,nbirths]);
        
        initial_state(1).x = [100;0; 3; -10];           tbirth(1) = 1;   tdeath(1) = K;
        initial_state(2).x = [400; -600; -10; 5];       tbirth(2) = 1;   tdeath(2) = K;
        initial_state(3).x = [-400; -200; 20; -5];      tbirth(3) = 1;   tdeath(3) = K;
        initial_state(4).x = [30; -200; 10.5; -5];      tbirth(4) = 1;   tdeath(4) = K;
        initial_state(5).x = [500; 200; -3; -15];       tbirth(5) = 1;   tdeath(5) = K;
end

%Generate true object data (noisy or noiseless) and measurement data
ground_truth = modelgen.groundtruth(nbirths,[initial_state.x],tbirth,tdeath,K);
objectdata = objectdatagen(ground_truth,motion_model,ifnoisy);
measdata = measdatagen(objectdata,sensor_model,meas_model);


% %build a TOMHT tracker
% numTracks = 20; % Maximum number of tracks
% gate = 40;      % Association gate
% vol = 1e2;      % Sensor bin volume
% beta = 1e-3;   % Rate of new targets in a unit volume
% pd = 0.8;       % Probability of detection
% far = 1e-3;     % False alarm rate
% tracker = trackerTOMHT( ...
%                         'FilterInitializationFcn',@initcvkf, ...
%                         'MaxNumTracks', numTracks, ...
%                         'MaxNumSensors', 1, ...
%                         'AssignmentThreshold', [0.6, 1, 1.2]*gate,...
%                         'DetectionProbability', pd, 'FalseAlarmRate', far, ...
%                         'Volume', vol, 'Beta', beta, ...
%                         'MaxNumHistoryScans', 6,...
%                         'MaxNumTrackBranches', 2,...
%                         'NScanPruning', 'Hypothesis', ...
%                         'OutputRepresentation', 'Tracks');
%               
% 
% 
% time = 0.5;
% wbar = waitbar(0,sprintf('Calculating HO-MHT iterations - k=%d',0));
% for i=1:100
%     for k=1:length(measdata{i,1})
%         dataLog.Detections{i}(:,k) = objectDetection(time, measdata{i,1}(:,k));
%     end
%     
%     tempdata =  measdata{i,1:end};
%     x = tempdata(1,:);
%     y =  tempdata(2,:);
%     plot(x, y, 'g.');hold on;
%       
%     tracks = tracker(dataLog.Detections{i}, time);
% 
%     time = time + 1;
%     dataLog.Time(i) = time;
% 
%     [pos,cov] = getTrackPositions(tracks,[1 0 0 0;0 0 1 0]);
%     plot(pos(:,1),pos(:,2),'r.');hold on
%     pause(0.01);
%     waitbar(i/100, wbar, sprintf('Calculating HO-MHT iterations - k=%d/%d',i,100));
% end
% close(wbar)

%N-object tracker parameter setting
P_G = 0.999;            %gating size in percentage
w_min = 1e-3;           %hypothesis pruning threshold
merging_threshold = 5;  %hypothesis merging threshold
M = 5;                 %maximum number of hypotheses kept in MHT
density_class_handle = feval(@GaussianDensity);    %density class handle
tracker = n_objectracker();
tracker = tracker.initialize(density_class_handle,P_G,meas_model.d,w_min,merging_threshold,M);

%GNN filter
[GNN_x,GNN_P] = GNNfilter(tracker, initial_state, measdata, sensor_model, motion_model, meas_model);
GNN_RMSE = RMSE_n_objects(objectdata.X,GNN_x);

%JPDA filter
[JPDA_x,JPDA_P] = JPDAfilter(tracker, initial_state, measdata, sensor_model, motion_model, meas_model);
JPDA_RMSE = RMSE_n_objects(objectdata.X,JPDA_x);

%Multi-hypothesis tracker
[TOMHT_x, TOMHT_P] = TOMHT(tracker, initial_state, measdata, sensor_model, motion_model, meas_model);
TOMHT_RMSE = RMSE_n_objects(objectdata.X,TOMHT_x);

X = sprintf('Root mean square error: GNN: %.3f; JPDA: %.3f; MHT: %.3f.',GNN_RMSE,JPDA_RMSE,TOMHT_RMSE);
disp(X)

% animate = Animate_2D_tracking();
% animate.animate(struct('x',TOMHT_x, 'P',TOMHT_P), initial_state, measdata, meas_model, range_c);

%Ploting
figure
hold on
grid on

% figure
x = [];
y = [];
for i=1:numel(measdata)
   tempdata =  measdata{i,1:end};
    x = [x tempdata(1,:)];
    y = [y tempdata(2,:)];
end
plot(x, y, 'r.')


for i = 1:nbirths
    h1 = plot(cell2mat(cellfun(@(x) x(1,i), objectdata.X, 'UniformOutput', false)), ...
        cell2mat(cellfun(@(x) x(2,i), objectdata.X, 'UniformOutput', false)), 'g-s', 'Linewidth', 1);
    h2 = plot(cell2mat(cellfun(@(x) x(1,i), GNN_x, 'UniformOutput', false)), ...
        cell2mat(cellfun(@(x) x(2,i), GNN_x, 'UniformOutput', false)), 'r-o', 'Linewidth', 1);
    h3 = plot(cell2mat(cellfun(@(x) x(1,i), JPDA_x, 'UniformOutput', false)), ...
        cell2mat(cellfun(@(x) x(2,i), JPDA_x, 'UniformOutput', false)), 'm-o', 'Linewidth', 1);
    h4 = plot(cell2mat(cellfun(@(x) x(1,i), TOMHT_x, 'UniformOutput', false)), ...
        cell2mat(cellfun(@(x) x(2,i), TOMHT_x, 'UniformOutput', false)), 'b-d', 'Linewidth', 1);
    h2.Color(4) = 0.3;
    h3.Color(4) = 0.3;
    h4.Color(4) = 0.3;
end

% xlim( [min(h1.XData), max(h1.XData)] );
% ylim( [min(h1.YData), max(h1.YData)] );

xlabel('x'); ylabel('y')

% legend([h1 h2 h3 h4],'Ground Truth','GNN','JPDA','TOMHT', 'Location', 'best')


set(gca,'FontSize',12) 
