function A2 = mapSpatialFeaturesLinear(A1, mask1, mask2)
%MAPSPATIALFEATURESLINEAR Map spatial footprints from mouse1 to mouse2
%
%   A2 = mapSpatialFeaturesLinear(A1, mask1, mask2)
%
%   Given:
%       A1    - (nPixels1 x K) spatial footprints for mouse 1
%       mask1 - (height1 x width1) binary mask (1=valid, 0=masked)
%       mask2 - (height2 x width2) binary mask
%
%   Returns:
%       A2    - (nPixels2 x K) mapped spatial footprints for mouse 2
%
%   This function assumes pure linear scaling between fields of view:
%   the normalized (u,v) coordinates [0,1]x[0,1] correspond between
%   the two mice. Masked pixels in mask2 will remain zero.

%% Validate inputs
[H1, W1] = size(mask1);
[H2, W2] = size(mask2);

[nPixels1, K] = size(A1);
if nPixels1 ~= H1*W1
    error('A1 must have height1*width1 rows');
end

% reshape A1 to 3D spatial maps
A1_maps = reshape(A1, H1, W1, K);

% Precompute normalized grids
[u1, v1] = meshgrid(linspace(0,1,W1), linspace(0,1,H1));  % Mouse 1 grid
[u2, v2] = meshgrid(linspace(0,1,W2), linspace(0,1,H2));  % Mouse 2 grid

% Initialize output
A2 = zeros(H2*W2, K);

% Loop over components
for k = 1:K
    % extract spatial map and apply mask1
    fmap1 = A1_maps(:,:,k);
    fmap1(~mask1) = 0;
    
    % create interpolant on normalized grid
    F = griddedInterpolant(u1', v1', fmap1', 'linear', 'nearest');
    
    % evaluate on mouse 2 normalized grid
    fmap2 = F(u2', v2')';
    
    % mask out invalid pixels
    fmap2(~mask2) = 0;
    
    % flatten
    A2(:,k) = fmap2(:);
    
    % optional normalization: each component has max = 1
    if any(A2(:,k) ~= 0)
        A2(:,k) = A2(:,k) / max(A2(:,k));
    end
end

end
