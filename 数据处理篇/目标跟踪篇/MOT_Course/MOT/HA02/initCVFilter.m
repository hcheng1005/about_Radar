function filter = initCVFilter(detection)
% initCVFilter - Initialize a constant velocity filter
% This function modifies the function |initcvekf| to handle higher velocity
% targets such as the airliners in the ATC scenario.
filter = initcvekf(detection);
classToUse = class(filter.StateCovariance);

% Airliners can move at speeds around 900 km/h. The velocity is
% initialized to 0, but will need to be able to quickly adapt to
% aircraft moving at these speeds. Use 900 km/h as 1 standard deviation
% for the initialized track's velocity noise.
spd = 900*1e3/3600; % m/s
velCov = cast(spd^2,classToUse);
cov = filter.StateCovariance;
cov(2,2) = velCov;
cov(4,4) = velCov;
filter.StateCovariance = cov;

% Increase the filter process noise to account for unknown acceleration.
scaleAccelHorz = cast(30,classToUse);
Gh = scaleAccelHorz;
Qh = Gh*Gh';
Q = blkdiag(Qh, Qh, 1);
filter.ProcessNoise = Q;
end