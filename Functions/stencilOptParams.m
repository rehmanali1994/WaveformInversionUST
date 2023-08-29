function [b,d,e] = stencilOptParams(vmin,vmax,f,h,g)
%STENCILOPTPARAMS Optimal Params for 9-Point Stencil 
%   INPUTS:
%       vmin = minimum wave velocity [L/T]
%       vmax = maximum wave velocity [L/T]
%       f = frequency [1/T]
%       h = grid spacing in X [L]
%       g = (grid spacing in Y [L])/(grid spacing in X [L])
%   OUTPUTS:
%       b, d, e = optimal params according to Chen/Cheng/Feng/Wu 2013 Paper

l = 100; r = 10;
Gmin = vmin/(f*h); Gmax = vmax/(f*h);

m = 1:l; n = 1:r;
theta = (m-1)*pi/(4*(l-1));
G = 1./(1/Gmax + ((n-1)/(r-1))*(1/Gmin-1/Gmax));

[TH, GG] = meshgrid(theta, G);

P = cos(g*2*pi*cos(TH)./GG);
Q = cos(2*pi*sin(TH)./GG);

S1 = (1+1/(g^2))*(GG.^2).*(1-P-Q+P.*Q);
S2 = (pi^2)*(2-P-Q);
S3 = (2*pi^2)*(1-P.*Q);
S4 = 2*pi^2 + (GG.^2).*((1+1/(g^2))*P.*Q-P-Q/(g^2));

fixB = true;
if fixB
    b = 5/6; % Fix the Value to 5/6 based on Laplacian Derived by Robert E. Lynch
    A = [S2(:), S3(:)]; y = S4(:)-b*S1(:);
    params = (A'*A)\(A'*y);
    d = params(1); e = params(2);
else
    A = [S1(:), S2(:), S3(:)]; y = S4(:);
    params = (A'*A)\(A'*y);
    b = params(1); d = params(2); e = params(3);
end

end

