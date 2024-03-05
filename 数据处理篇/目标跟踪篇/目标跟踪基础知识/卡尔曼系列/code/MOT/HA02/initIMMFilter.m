function filter = initIMMFilter(detection)
% initIMMFilter  Initialize an IMM filter with constant velocity and constant turn models

filter1 = initcvekf(detection);
filter2 = initctekf(detection);
filter = trackingIMM({filter1,filter2},'TransitionProbabilities',0.97);
classToUse = class(filter.StateCovariance);

% The velocity is initialized to 0, but will need to be able to quickly
% adapt to targets moving at 300 km/h. Use 900 km/h as 1 standard deviation
% for the initialized track's velocity noise.
spd = 900*1e3/3600; % m/s
velCov = cast(spd^2,classToUse);
scaleAccelHorz = cast(10,classToUse); % Standard deviation for the horizontal acceleration
scaleOmegaDot = cast(30,classToUse);  % Standard deviation for the turn rate change

for i = 1:numel(filter.TrackingFilters)
    filter.TrackingFilters{i}.StateCovariance(2:2:4,2:2:4) = blkdiag(velCov,velCov);
    Gh = scaleAccelHorz;
    Qh = Gh*Gh';
    filter.TrackingFilters{i}.ProcessNoise(1:2,1:2) = blkdiag(Qh, Qh);
    if strcmpi(func2str(filter.TrackingFilters{i}.StateTransitionFcn),'constturn')
        Qo = scaleOmegaDot^2;
        Q(1:2,1:2) = blkdiag(Qh, Qh);
        Q(3,3) = Qo;
    end
end
end