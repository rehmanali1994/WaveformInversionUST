function [C, c_bkgnd] = soundSpeedPhantom2D(Xi, Yi)
%SOUNDSPEEDPHANTOM Outputs Sound Speed Phantom
% [C, c_bkgnd] = soundSpeedPhantom(Xi, Yi, option)
% INPUTS:
%   Xi, Yi -- meshgrid of points over which sound speed is defined
%   option -- 1 for breast CT; 2 for breast MRI; 3 for skull CT
% OUTPUTS:
%   C -- sound speed map on grid [m/s]
%   c_bkgnd -- background sound speed [m/s]

% Load Breast CT Image
[~, E] = phantom('Modified Shepp-Logan',1001);
E(:,1) = [1,-0.5,-0.5,-0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25];
shepplogan = phantom(E,1001);

% Get Dimensions of Breast CT Image
[Ny, Nx] = size(shepplogan); 
dx = 0.00015; dy = dx; % Grid Spacing [m]
x = ((-(Nx-1)/2):((Nx-1)/2))*dx; 
y = ((-(Ny-1)/2):((Ny-1)/2))*dy;
[X, Y] = meshgrid(x,y); 
% Create Sound Speed Image [m/s]
c_min = 1500; c_max = 1580;
c = shepplogan*(c_max-c_min)+c_min;
c_bkgnd = c(1,1);
% Put Sound Speed Map on Input Meshgrid
R = sqrt(Xi.^2 + Yi.^2); 
rotAngle = 0; % Angle [radians] to Rotate the Breast 
T = atan2(Yi, Xi) - rotAngle; % Apply Rotation
C = interp2(X, Y, c, R.*cos(T), R.*sin(T), 'linear', c_bkgnd);

end

