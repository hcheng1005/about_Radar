function measdata = measdatagen(objectdata, sensormodel, measmodel)
%MEASDATAGEN generates object-generated measurements and clutter
%INPUT:     objectdata: a structure contains object data
%           sensormodel: a structure specifies sensor model parameters
%           measmodel: a structure specifies the measurement model
%           parameters 
%OUTPUT:    measdata: cell array of size (total tracking time, 1), each
%           cell stores measurements of size (measurement dimension) x
%           (number of measurements at corresponding time step) 

%Initialize memory
measdata = cell(length(objectdata.X),1);

%Generate measurements
for k = 1:length(objectdata.X)
%     if objectdata.N(k) > 0
%         idx = rand(objectdata.N(k),1) <= sensormodel.P_D;
%         %Only generate object-originated observations for detected objects
%         if ~isempty(objectdata.X{k}(:,idx))
%             objectstates = objectdata.X{k}(:,idx);
%             for i = 1:size(objectstates,2)
%                 meas = mvnrnd(measmodel.h(objectstates(:,i))', measmodel.R)';
%                 measdata{k} = [measdata{k} meas];
%             end
%         end
%     end
    
    idx = [1:1:objectdata.N(k)];
%     idx = rand(objectdata.N(k),1) <= sensormodel.P_D;

        objectstates = objectdata.X{k}(:,idx);
        for i = 1:size(objectstates,2)
            meas = measmodel.h(objectstates(:,i));
            measdata{k} = [measdata{k} meas];
        end
         
                
    [a,~] = find(idx > 0); 
    for i= 1:length(a)
        
        orignaldata = measdata{k,1}(:,i);
        
        %Number of clutter measurements
        N_c = poissrnd(sensormodel.lambda_c);
        
        R = 20;
        rand1 = R*rand(1, N_c) - 0.5 *  R;
        rand2 = R*rand(1, N_c) - 0.5 *  R;
        C = repmat(orignaldata,[1 N_c]) + reshape([rand1 rand2],[2, N_c]);

        %Generate clutter
%         C = repmat(sensormodel.range_c(:,1),[1 N_c])+ diag(sensormodel.range_c*[-1; 1])*rand(measmodel.d,N_c);
        %Total measurements are the union of object detections and clutter
        measdata{k}= [measdata{k} C]; 
    end
end

end