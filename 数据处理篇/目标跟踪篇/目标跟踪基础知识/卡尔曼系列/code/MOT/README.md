# MOT介绍


# Multi-Object-Tracking
Repository for the course "Multi-Object Tracking for Automotive Systems" at EDX Chalmers University of Technology

YouTube: https://www.youtube.com/channel/UCa2-fpj6AV8T6JK1uTRuFpw

## Home-Assignment 01 (HA01) - Single-Object Tracking in Clutter
Implementation of the following algorithms:
- [x] Nearest Neighbors Filter (NN)
- [x] Probabilistic Data Association Filter (PDA)
- [x] Gaussian Sum Filter (GSF)

The main class is located at [HA01/singleobjectracker.m](./HA01/singleobjectracker.m)

Simulations can be done using either [HA01/main.m](./HA01/main.m) or [HA01/simulation.m](./HA01/simulation.m)

```matlab
%Nearest neighbour filter
nearestNeighborEstimates = nearestNeighbourFilter(tracker, initial_state, measdata, sensor_model, motion_model, meas_model);
nearestNeighborRMSE = RMSE(nearestNeighborEstimates,objectdata.X);

%Probabilistic data association filter
probDataAssocEstimates = probDataAssocFilter(tracker, initial_state, measdata, sensor_model, motion_model, meas_model);
probDataAssocRMSE = RMSE(probDataAssocEstimates,objectdata.X);

%Gaussian sum filter
GaussianSumEstimates = GaussianSumFilter(tracker, initial_state, measdata, sensor_model, motion_model, meas_model);
GaussianSumRMSE = RMSE(GaussianSumEstimates,objectdata.X);
```

## Home-Assignment 02 (HA02) - Tracking n Objects in Clutter
Implementation of the following algorithms:
- [x] Global Nearest Neighbors Filter (GNN)
- [x] Joint Probabilistic Data Association Filter (JPDA)
- [x] Track-oriented Multiple Hypothesis Tracker (TO-MHT)

The main class is located at [HA02/n_objectracker.m](./HA02/n_objectracker.m)

Simulations can be done using either [HA02/main.m](./HA02/main.m) or [HA02/simulation.m](./HA02/simulation.m)

```matlab
%GNN filter
[GNN_x,GNN_P] = GNNfilter(tracker, initial_state, measdata, sensor_model, motion_model, meas_model);
GNN_RMSE = RMSE_n_objects(objectdata.X,GNN_x);

%JPDA filter
[JPDA_x,JPDA_P] = JPDAfilter(tracker, initial_state, measdata, sensor_model, motion_model, meas_model);
JPDA_RMSE = RMSE_n_objects(objectdata.X,JPDA_x);

%Multi-hypothesis tracker
[TOMHT_x, TOMHT_P] = TOMHT(tracker, initial_state, measdata, sensor_model, motion_model, meas_model);
TOMHT_RMSE = RMSE_n_objects(objectdata.X,TOMHT_x);
```

## Home-Assignment 03 (HA03) - Random Finite Sets
- [ ] Probability Hypothesis Density Filter (PHD)
- [ ] Gaussian Mixture Probability Hypothesis Density Filter (GM-PHD)

## Home-Assignment 04 (HA04) - MOT Using Conjugate Priors
- [ ] Multi-Bernoulli Mixture filter (MBM)
- [ ] Poisson Multi-Bernoulli Mixture filter (PMBM)

## Home-Assigment 05 (HA05) - Extended Object Tracking
