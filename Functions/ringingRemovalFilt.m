function img_out = ringingRemovalFilt(xin, yin, img_in, c0, f, cutoff, ord)
%RINGINGREMOVALFILT Filter to Remove Ringing from Waveform Inversion Result
%
% img_out = ringingRemovalFilt(xin, yin, img_in, c0, f, cutoff)
% INPUT:
%   xin = x (N-element array) grid [in m] for the input image
%   yin = y (M-element array) grid [in m] for the input image
%   img_in = input image (M x N array)
%   c0 = reference sound speed [m/s]
%   f = frequency [Hz]
%   cutoff = ranging from 0 (only keep DC) to 1 (up to ringing frequency)
%   ord = order of radial Butterworth filter cutoff (Inf for sharp cutoff)
% OUTPUT:
%   img_out = output image (M x N array)

% Number and spacing of input points (assumes a uniform spacing)
Nxin = numel(xin); dxin = mean(diff(xin));
Nyin = numel(xin); dyin = mean(diff(yin));

% K-Space
kxin = fftshift(((0:Nxin-1)/Nxin)/dxin);
kxin(kxin>=1/(2*dxin)) = kxin(kxin>=1/(2*dxin)) - 1/dxin;
kyin = fftshift(((0:Nyin-1)/Nyin)/dyin);
kyin(kyin>=1/(2*dyin)) = kyin(kyin>=1/(2*dyin)) - 1/dyin;
[Kxin, Kyin] = meshgrid(kxin, kyin);

% Cutoff Wavenumber
k = 2*f/c0; kcutoff = k*cutoff;

% Filter Based on Cutoff
radialFilt = 1./(1+((Kxin.^2 + Kyin.^2)/(kcutoff^2)).^ord);

% Apply Filter
img_out = real(ifft2(ifftshift(radialFilt.*(fftshift(fft2(img_in))))));

end