clear
clc

% Add Functions to Path
addpath(genpath('Functions'));

% Load Saved Results
filename = 'Malignancy'; % Could be: 'kWave_BreastCT', 'kWave_BreastMRI'
    % 'VSX_YezitronixPhantom1', 'VSX_YezitronixPhantom2', 'BenignCyst', 'Malignancy' 
filename_results = ['Results/', filename, '_WaveformInversionResults.mat'];
load(filename_results, 'xi', 'yi', 'fDATA', 'niterAttenPerFreq', ...
    'niterSoSPerFreq', 'VEL_ESTIM_ITER', 'ATTEN_ESTIM_ITER', 'GRAD_IMG_ITER', 'SEARCH_DIR_ITER')

% Display Constants
crange = [1350, 1600]; % For reconstruction display [m/s]
attenrange = 10*[-1,1]; % For reconstruction display [dB/(cm MHz)]

% Conversion of Units for Attenuation Map
Np2dB = 20/log(10);
slow2atten = (1e6)/(1e2); % Hz to MHz; m to cm

%% Visualize Numerical Solution
for f_idx = 1:numel(fDATA)
    % Iterations at Each Frequency
    for iter_f_idx = 1:(niterSoSPerFreq(f_idx)+niterAttenPerFreq(f_idx))
        iter = iter_f_idx + sum(niterSoSPerFreq(1:f_idx-1)) + ...
            sum(niterAttenPerFreq(1:f_idx-1));
        subplot(2,2,1); imagesc(xi,yi,VEL_ESTIM_ITER(:,:,iter),crange);
        title(['Estimated Wave Velocity ', num2str(iter)]); axis image;
        xlabel('Lateral [m]'); ylabel('Axial [m]'); colorbar; colormap gray;
        subplot(2,2,2); imagesc(xi,yi,Np2dB*slow2atten*ATTEN_ESTIM_ITER(:,:,iter),attenrange);
        title(['Estimated Attenuation ', num2str(iter)]); axis image;
        xlabel('Lateral [m]'); ylabel('Axial [m]'); colorbar; colormap gray;
        subplot(2,2,3); imagesc(xi,yi,SEARCH_DIR_ITER(:,:,iter));
        xlabel('Lateral [m]'); ylabel('Axial [m]'); axis image;
        title(['Search Direction Iteration ', num2str(iter)]); colorbar; colormap gray; 
        subplot(2,2,4); imagesc(xi,yi,-GRAD_IMG_ITER(:,:,iter));
        xlabel('Lateral [m]'); ylabel('Axial [m]'); axis image;
        title(['Gradient Iteration ', num2str(iter)]); colorbar; colormap gray; 
        drawnow; disp(['Iteration ', num2str(iter)]);
        disp(['Frequency ', num2str((1e-3)*fDATA(f_idx)), ' kHz']);
        if iter_f_idx <= niterSoSPerFreq(f_idx)
            disp('Sound Speed Update');
        else
            disp('Attenuation Update')
        end
    end
end