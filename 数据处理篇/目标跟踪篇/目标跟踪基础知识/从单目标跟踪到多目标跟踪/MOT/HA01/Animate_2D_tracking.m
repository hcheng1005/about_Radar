classdef Animate_2D_tracking
    %ANIMATE_2D_TRACKING Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
    end
    
    methods(Static)
        function [ xy ] = sigmaEllipse2D( mu, Sigma, level, npoints )
            %SIGMAELLIPSE2D generates x,y-points which lie on the ellipse describing
            % a sigma level in the Gaussian density defined by mean and covariance.
            %
            %Input:
            %   MU          [2 x 1] Mean of the Gaussian density
            %   SIGMA       [2 x 2] Covariance matrix of the Gaussian density
            %   LEVEL       Which sigma level curve to plot. Can take any positive value, 
            %               but common choices are 1, 2 or 3. Default = 3.
            %   NPOINTS     Number of points on the ellipse to generate. Default = 32.
            %
            %Output:
            %   XY          [2 x npoints] matrix. First row holds x-coordinates, second
            %               row holds the y-coordinates. First and last columns should 
            %               be the same point, to create a closed curve.

            %Setting default values, in case only mu and Sigma are specified.
            if nargin < 3
                level = 3;
            end
            if nargin < 4
                npoints = 32;
            end

            % Procedure:
            % - A 3 sigma level curve is given by {x} such that (x-mux)'*Q^-1*(x-mux) = 3^2
            %      or in scalar form: (x-mux) = sqrt(Q)*3
            % - replacing z= sqrtm(Q^-1)*(x-mux), such that we have now z'*z = 3^2
            %      which is now a circle with radius equal 3.
            % - Sampling in z, we have z = 3*[cos(theta); sin(theta)]', for theta=1:2*pi
            % - Back to x we get:  x = mux  + 3* sqrtm(Q)*[cos(theta); sin(theta)]'

            xy = [];
            for ang = linspace(0,2*pi,npoints)
                xy(:,end+1) = mu + level * sqrtm(Sigma) * [cos(ang) sin(ang)]';
            end
        end
        
        function [zk,S] = estimated_meas(x,P,measmodel)
            % measurement covariance
            Hx = measmodel.H(x);
            % Innovation covariance
            S = Hx * P * Hx' + measmodel.R;
            % ensure it is positive definite
            S = (S+S')/2;
            % object measurement
            zk = measmodel.h(x);
        end
    end
    
    methods
        function obj = animate_2D_tracking(obj)
        end
        
        function animate(obj, est, initial_state, measdata, measmodel, range_c)
            fig = figure;
            hold on;
            xlim(range_c(1,:));
            ylim(range_c(2,:));
            
            % plot meas data
            meas_x = measdata{1}(1,:);
            meas_y = measdata{1}(2,:);
            pl_meas = scatter(meas_x, meas_y, 50, 'o', 'filled');
            pl_meas.XDataSource = 'meas_x';
            pl_meas.YDataSource = 'meas_y';
            
            [zk,S] = obj.estimated_meas(initial_state.x,initial_state.P,measmodel);
            
            % plot estimated covariance
            ellipse = obj.sigmaEllipse2D(zk, S);
            S_x = ellipse(1,:); 
            S_y = ellipse(2,:); 
            pl_cov  = plot(S_x,S_y);
            pl_cov.XDataSource = 'S_x'; 
            pl_cov.YDataSource = 'S_y';
            
            % plot estimated mean
            zk_x = zk(1);
            zk_y = zk(2);
            pl_mean = scatter(zk_x, zk_y,100, 'o', 'filled');
            pl_mean.XDataSource = 'zk_x';
            pl_mean.YDataSource = 'zk_y';
            
            for i=1:numel(measdata)
                [zk,S] = obj.estimated_meas(est(i).x,est(i).P,measmodel);
                ellipse = obj.sigmaEllipse2D(zk, S);
                % variance
                S_x = ellipse(1,:); 
                S_y = ellipse(2,:);
                % mean
                zk_x = zk(1);
                zk_y = zk(2);
                % meas
                meas_x = measdata{i}(1,:);
                meas_y = measdata{i}(2,:);
                
                refreshdata(fig,'caller');
                drawnow;
%                 pause(0.05);
            end
        end
        
    end
end





