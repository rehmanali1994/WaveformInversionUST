function [x_idx, z_idx, ind] = sampled_circle(Nx, Nz, Nr, theta)
%SAMPLED CIRCLE Creates a sample circle mask
%   Nx, Nz -- Size of x and z grids
%   Nr -- Radius of circle in samples
%   theta -- Sampled angles

x = (-(Nx-1)/2:(Nx-1)/2);
z = (-(Nz-1)/2:(Nz-1)/2);
x_idx = dsearchn(x(:), Nr*cos(theta(:)));
z_idx = dsearchn(z(:), Nr*sin(theta(:)));
ind = sub2ind([Nz, Nx], z_idx, x_idx);

end

