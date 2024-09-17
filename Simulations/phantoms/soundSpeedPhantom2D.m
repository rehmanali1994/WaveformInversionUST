function [C, c_bkgnd] = soundSpeedPhantom2D(Xi, Yi, option)
%SOUNDSPEEDPHANTOM Outputs Sound Speed Phantom
% [C, c_bkgnd] = soundSpeedPhantom(Xi, Yi, option)
% INPUTS:
%   Xi, Yi -- meshgrid of points over which sound speed is defined
%   option -- 1 for breast CT; 2 for breast MRI; 3 for skull CT
% OUTPUTS:
%   C -- sound speed map on grid [m/s]
%   c_bkgnd -- background sound speed [m/s]

switch option
    case 1 % Breast CT Phantom
        % Load Breast CT Image
        breastct = im2double(imread('breast_ct.jpg'));
        % Normalize Breast CT Image
        breastct = breastct/max(breastct(:)); thr = 0.04;
        breastct(breastct<=thr) = mean(breastct(breastct>=thr));
        breastct = breastct - mean(breastct(:));
        breastct = breastct/max(abs(breastct(:)));
        % Get Dimensions of Breast CT Image
        [Ny, Nx] = size(breastct); 
        dx = 0.0007; dy = dx; % Grid Spacing [m]
        x = ((-(Nx-1)/2):((Nx-1)/2))*dx; 
        y = ((-(Ny-1)/2):((Ny-1)/2))*dy;
        [X, Y] = meshgrid(x,y); 
        % Create Sound Speed Image [m/s]
        c_bkgnd = 1500; c_std = 90;
        c = c_bkgnd+c_std*breastct;
        % Put Sound Speed Map on Input Meshgrid
        R = sqrt(Xi.^2 + Yi.^2); 
        rotAngle = 2.85*pi; % Angle [radians] to Rotate the Breast 
        T = atan2(Yi, Xi) - rotAngle; % Apply Rotation
        C = interp2(X, Y, c, R.*cos(T), R.*sin(T), 'linear', c_bkgnd);
    case 2 % Breast MRI Phantom
        % Load Breast CT Image
        breastmri = im2double(imread('breast_mri.jpg'));
        % Get Dimensions of Breast CT Image
        [Ny, Nx] = size(breastmri); 
        dx = 0.00015; dy = dx; % Grid Spacing [m]
        x = ((-(Nx-1)/2):((Nx-1)/2))*dx; 
        y = ((-(Ny-1)/2):((Ny-1)/2))*dy;
        [X, Y] = meshgrid(x,y); 
        % Create Sound Speed Image [m/s]
        c_min = 1420; c_max = 1640;
        c = breastmri*(c_max-c_min)+c_min;
        c_bkgnd = c(1,1);
        % Put Sound Speed Map on Input Meshgrid
        R = sqrt(Xi.^2 + Yi.^2); 
        rotAngle = 2.85*pi; % Angle [radians] to Rotate the Breast 
        T = atan2(Yi, Xi) - rotAngle; % Apply Rotation
        C = interp2(X, Y, c, R.*cos(T), R.*sin(T), 'linear', c_bkgnd);
end

end

