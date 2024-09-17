clear
clc

% Load Functions
addpath(genpath('../../k-Wave'));
addpath(genpath(pwd));

% Sound Speed Map 
dxi = 0.15e-3; xmax = 120e-3;
xi = -xmax:dxi:xmax; zi = xi;
Nxi = numel(xi); Nzi = numel(zi);
[Xi, Zi] = meshgrid(xi, zi);
[C, c_bkgnd] = soundSpeedPhantom2D(Xi, Zi);

% Attenuation Map
atten_bkgnd = 0.0025; % Background Attenuation [dB/(MHz^y cm)]
sos2atten = 1.5e-2; % Conversion from Sound Speed Difference to Attenuation
atten_varying = sos2atten*abs(C-c_bkgnd); % Varying Attenuation [dB/(MHz^y cm)]
atten = (atten_bkgnd + atten_varying); % Total Attenuation [dB/(MHz^y cm)]
y_atten = 1.01; % Power Law of Attenuation [y]
    % y cannot exactly equal 1 without the following line:
    % medium.alpha_mode = 'no_dispersion'

% Create Transducer Ring
circle_radius = 110e-3; numElements = 512;
circle_rad_pixels = floor(circle_radius/dxi);
theta = -pi:2*pi/numElements:pi-2*pi/numElements;
x_circ = circle_radius*cos(theta); 
z_circ = circle_radius*sin(theta); 
[x_idx, z_idx, ind] = sampled_circle(Nxi, Nzi, circle_rad_pixels, theta);
msk = zeros(Nzi, Nxi); msk(ind) = 1;

% Define the Properties of the Propagation Medium
rho0 = 1000; % Density [kg/m^3]
medium.sound_speed = C; % [m/s]
medium.density = rho0*ones(size(C)); % [kg/m^3]
medium.alpha_coeff = atten; % [dB/(MHz^y cm)]
medium.alpha_power = y_atten; % cannot exactly equal 1 without:
                              % medium.alpha_mode = 'no_dispersion'
medium.alpha_mode = 'no_dispersion'; % IGNORE VELOCITY DISPERSION!
kgrid = kWaveGrid(Nzi, dxi, Nxi, dxi); % K-Space Grid Object

% Create Time Array
t_end = 1.3 * Nzi * dxi / min(C(:)); cfl = 0.3;
[kgrid.t_array, dt] = makeTime(kgrid, C, cfl, t_end);
fs = 1/dt; % Sampling Frequency [Hz]

% Define Properties of the Tone Burst Used to Drive the Transducer

% Transmit Pulse and Impulse Response
fracBW = 0.75;  % Fractional Bandwidth of the Transducer
fTx = 1.0e6; % Transmit Frequency [Hz]
tc = gauspuls('cutoff', fTx, fracBW, -6, -80); % Cutoff time at -80dB, fracBW @ -6dB
t = (-ceil(tc/dt):1:ceil(tc/dt))*dt; % (s) Time Vector centered about t=0
src_amplitude = 1e1;
tx_signal = src_amplitude * gauspuls(t, fTx, fracBW); % Calculate Transmit Pulse
[~, emission_samples] = size(tx_signal); 
t_offset = -(emission_samples-1)*dt/2; 

% Define a Binary Sensor Mask
sensor.mask = zeros(Nzi, Nxi);
sensor.mask(ind) = 1; % Make Sensor the Same As Source

% Create a Display Mask to Display the Transducer
display_mask = sensor.mask;

% Assign the Input Options
input_args = {'DisplayMask', display_mask, 'PMLInside', false, ...
    'PlotPML', false, 'PMLAlpha', 10, 'PlotSim', false, 'DataCast', 'gpuArray-single'};

% Save Simulation Info to File
save('sim_info/SimInfo.mat');