function scenario = helperCreateScenario
%% Closely Spaced Targets Scenario
% Simulate a scanning radar mounted on a tower and the motion of the targets
% in the scenario as _platforms_. Simulation of the motion of the platforms in
% the scenario is managed by |trackingScenario|.
%
% Create a |trackingScenario| and add the tower to the scenario.

% Create tracking scenario
scenario = trackingScenario;

%% Scanning Radar
% Add a scanning radar mounted 15 meters above the ground. This radar scans
% mechanically in azimuth at a fixed rate to provide 360-degree coverage in
% the vicinity of the mounting platform. Specifications for this scanning
% radar were taken to mimic an Airport Surveillance Radar and are listed
% below:
% 
% * Sensitivity:            0 dBsm @ 111 km
% * Mechanical Scan:        Azimuth only
% * Mechanical Scan Rate:   12.5 RPM
% * Electronic Scan:        None
% * Field of View:          1.5 deg azimuth, 10 deg elevation
% * Azimuth Resolution:     1.5 deg
% * Range Resolution:       135 m
% 
% Model the scanning radar with the above specifications using the |monostaticRadarSensor|.

rpm = 25;
fov = [1.5;10];
scanrate = rpm*360/60;  % deg/s
updaterate = scanrate/fov(1); % Hz
pd = 0.8;
far = 1e-6;

radar = monostaticRadarSensor(1,'Rotator', ...
    'UpdateRate', updaterate, ...           % Hz
    'FieldOfView', fov, ...                 % [az;el] deg
    'MaxMechanicalScanRate', scanrate, ...  % deg/sec
    'AzimuthResolution', fov(1), ...        % deg
    'ReferenceRange', 111e3, ...            % m
    'ReferenceRCS', 0, ...                  % dBsm
    'RangeResolution', 135, ...             % m
    'HasINS', true, ...
    'MechanicalScanLimits', [250 290], ...
    'DetectionCoordinates', 'Scenario',...
    'DetectionProbability', pd, ...
    'FalseAlarmRate', far);

% Mount radar 15 meters high on a platform
radar.MountingLocation = [0 0 -15];
platform(scenario, 'Sensors', radar);

%%
% Tilt the radar so that it surveys a region beginning at 2 degrees above
% the horizon. To do this, enable elevation and set the mechanical scan
% limits to span the radar's elevation field of view beginning at 2 degrees
% above the horizon. Because |trackingScenario| uses a North-East-Down
% (NED) coordinate frame, negative elevations correspond to points above
% the horizon.

% Enable elevation scanning
radar.HasElevation = true;

% Set mechanical elevation scan to begin at 2 degrees above the horizon
elFov = fov(2);
tilt = 2; % deg
radar.MechanicalScanLimits(2,:) = [-fov(2) 0]-tilt; % deg

%%
% Set the elevation field of view to be slightly larger than the elevation
% spanned by the scan limits. This prevents raster scanning in elevation
% and tilts the radar to point in the middle of the elevation scan limits.

radar.FieldOfView(2) = elFov+1e-3;

%%
% The |monostaticRadarSensor| models range and elevation bias due to atmospheric
% refraction. These biases become more pronounced at lower altitudes and
% for targets at long ranges. Because the index of refraction changes
% (decreases) with altitude, the radar signals propagate along a curved
% path. This results in the radar observing targets at altitudes which are
% higher than their true altitude and at ranges beyond their line-of-sight
% range.

%%
% Add two targets within the surveillance area. At the begining of the
% scenario both targets are moving along the x axis towards each other at
% constant velocity: 300km/h.
% After 15 seconds both targets make a 90 degrees 3G turn towards the positive
% y axis direction. They fly close to each other for 15 seconds before
% making another 90 degrees 3G turn to leave in the x axis direction, both
% returning in their direction of origin.
% |trackingScenario| uses a North-East-Down (NED) coordinate frame. When
% defining the waypoints for the targets below, the z-coordinate
% corresponds to down, so heights above the ground are set to negative
% values.

% Duration of scenario
sceneDuration = 50; % s

% Actor 1
ht = 3e3;
spd = 300*1e3/3600;     % m/s
accel = 3*9.8;          % m/s^2
radius = spd^2/accel;   % m
t0 = 0;
t1 = t0+15;
t2 = t1+pi/2*radius/spd;
t3 = t2+15;
t4 = t3+pi/2*radius/spd;
t5 = sceneDuration;
timeOfArrival = [t0 t1 t2 t3 t4 t5]';
course = [0,0,90,90,180,180];
vel = [spd*[cosd(course);sind(course)]',zeros(numel(course),1)];
wp = [ ...
    0                   0                       0; ... % Begin straight segment
    spd*t1              0                       0; ... % Begin turn at 3 G
    spd*t1+radius       radius                  0; ... % End of turn
    spd*t1+radius       radius+spd*(t3-t2)      0; ... % End of second straight segment
    spd*t1              2*radius+spd*(t3-t2)    0; ... % End of second turn
    spd*(t1+t4-t5)      2*radius+spd*(t3-t2)    0];... % End of third straight segement    
wp = wp+[-0.12e4-radius -1.995e4 -ht];
tgt = platform(scenario);
tgt.Trajectory = waypointTrajectory('Waypoints',wp,'TimeOfArrival',timeOfArrival,'Velocities', vel);

% Actor 2
ht = 3e3;
spd = 300*1e3/3600;     % m/s
accel = 3*9.8;          % m/s^2
radius = spd^2/accel;   % m
t0 = 0;
t1 = t0+15;
t2 = t1+pi/2*radius/spd;
t3 = t2+15;
t4 = t3+pi/2*radius/spd;
t5 = sceneDuration;
timeOfArrival = [t0 t1 t2 t3 t4 t5]';
course = [180,180,90,90,0,0];
vel = [spd*[cosd(course);sind(course)]',zeros(numel(course),1)];
wp = [ ...
    2*spd*(t1-t0)+radius        0                       0; ... % Begin straight segment
    spd*t1+radius               0                       0; ... % Begin turn at 3 G
    spd*t1                      radius                  0; ... % End of turn
    spd*t1                      radius+spd*(t3-t2)      0; ... % End of second straight segment
    spd*t1+radius               2*radius+spd*(t3-t2)    0; ... % End of second turn
    spd*(t1+t5-t4)+radius       2*radius+spd*(t3-t2)    0];... % End of third straight segement    
wp = wp+[-0.0996e4 -1.995e4 -ht];
tgt = platform(scenario);
tgt.Trajectory = waypointTrajectory('Waypoints',wp,'TimeOfArrival',timeOfArrival,'Velocities', vel);

% Set simulation to advance at the update rate of the radar
scenario.UpdateRate = radar.UpdateRate;