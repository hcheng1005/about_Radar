function plotScenarioAndDetections(dataLog)
% helper to draw a data log of a tracking scenario
% the data log should contain the true position of the track objects and the
% detections received at each time

hfig = figure;
hfig.Position =  [614   365   631   529];
hfig.Visible = 'on';
hfig.Color = [1 1 1];
tpaxes = axes(hfig);
grid(tpaxes,'on')
title(tpaxes,'Targets and detections')
tp = theaterPlot('Parent',tpaxes,'AxesUnits',["km" "km" "km"],'XLimits',[-2000 2000], 'YLimits',[-20500 -17000]);
detp = detectionPlotter(tp,'DisplayName','Detections','MarkerSize',6,'MarkerFaceColor',[0.85 0.325 0.098],'MarkerEdgeColor','k','History',1000);
platp = platformPlotter(tp,'DisplayName','Targets','MarkerFaceColor','k');
trajp = trajectoryPlotter(tp,'DisplayName','Trajectory');
% move legend inside
hfig.Children(1).Location = "northeast";

% plot full target trajectories
trajectory{1} = vertcat(dataLog.Truth(1,:).Position);
trajectory{2} = vertcat(dataLog.Truth(2,:).Position);
trajp.plotTrajectory(trajectory);

% update plot at each time update
for t=1:numel(dataLog.Time)
    % update detection position
    detections = dataLog.Detections{t};
    allDets = [detections{:}];
    meas = cat(2,allDets.Measurement);
    measCov = cat(3,allDets.MeasurementNoise);
    detp.plotDetection(meas',measCov);
    % update position of truth
    truePos = vertcat(dataLog.Truth(:,t).Position);
    platp.plotPlatform(truePos);
    drawnow
end
hold on
plot([-0.5;0.5]*1e3,[-19.8 -19.8]*1e3,':k')
plot([-0.5;0.5]*1e3,[-19.5 -19.5]*1e3,':k')
text(0.35*1e3,-19.65*1e3,'Ambiguity Region')
pause(2)
end