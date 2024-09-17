clear
clc

% Load Simulation Information Created by GenKWaveSimInfo.m
load('sim_info/SimInfo.mat');

% Assemble RF Data From Individual Tx Beams Into Full Synthetic Aperture
addpath(genpath('../k-Wave'));
dwnsmp = 5; % Subsample in Time
full_dataset = zeros(numel(kgrid.t_array(1:dwnsmp:end)), ...
    numElements, numElements, 'single');
for tx_elmt_idx = 1:numElements
    filename = ['scratch/rf_data_tx_elem_', num2str(tx_elmt_idx), '.mat'];
    load(filename, 'rf_data'); % Load RF Data From This Transmit Beam
    full_dataset(:, :, tx_elmt_idx) = single(rf_data(1:dwnsmp:end,:)); % Assemble
    disp(['Assembled ' num2str(tx_elmt_idx), ' Transmit Beam']);
end
time = kgrid.t_array(1:dwnsmp:end)+t_offset; 
transducerPositionsXY = [x_circ; z_circ];
xi_orig = xi; yi_orig = zi; 
clearvars -except option time full_dataset transducerPositionsXY xi_orig yi_orig C atten; 

% Save Full Synthetic Aperture Data
save('datasets/kWave_SheppLogan.mat', '-v7.3', ...
    'time', 'full_dataset', 'transducerPositionsXY', ...
    'xi_orig', 'yi_orig', 'C', 'atten');