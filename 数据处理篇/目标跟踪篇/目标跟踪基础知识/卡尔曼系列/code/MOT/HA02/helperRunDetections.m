function dataLog = helperRunDetections(scenario)
%helperRunDetections  Run the example scenario and create detections
% detLog = helperRunDetections(scenario) runs the scenario and returns a
% log of time-synchronized truth and detection data, dataLog.

%   Copyright 2018 The MathWorks, Inc.

% Get the tower and radar information from the scenario
tower = scenario.Platforms{1};
radar = tower.Sensors{1};


%% Simulate and Track Airliners
% The following loop advances the platform positions until the end of the
% scenario has been reached. For each step forward in the scenario, the
% radar generates detections from targets in its field of view. The tracker
% is updated with these detections after the radar has completed a 360
% degree scan in azimuth.

% Restart the scenario
restart(scenario);

% Create a buffer to collect the detections from a full scan of the radar
scanBuffer = {};
dataLog = struct('Time',{{}}, 'Truth', {{}}, 'Detections', {{}});
dataLog.Time = [];
dataLog.Truth = [];
dataLog.Detections = {};

% Set random seed for repeatable results
s = rng;
rng(2018)
disp('Please wait. Generating detections for scenario .....')
while advance(scenario)
    
    % Current simulation time
    simTime = scenario.SimulationTime;
    
    % Target poses in the ATC's coordinate frame
    targets = targetPoses(tower);
    
    % Use the tower's true position as its INS measurement
    ins = pose(tower, 'true');
    
    % Generate detections on target's in the radar's current field of view
    [dets,~,config] = radar(targets,ins,simTime);
    
    scanBuffer = [scanBuffer;dets]; %#ok<AGROW>
    
    % Update tracks when a 360 degree scan is complete
    if config.IsScanDone
        % Log the detections
        dataLog.Time = [dataLog.Time, simTime];
        dataLog.Truth = [dataLog.Truth, targets];
        dataLog.Detections = [dataLog.Detections(:)', {scanBuffer}];
        scanBuffer = {};
    end
end

% Return the random number generator to its initial condition
rng(s)
disp('Detections generation complete.')
end